/*
  tp0515.js（基于 tp0509.js）
  青藏高原湿地四类分类（10m, S2/S1, 双时相）
  ------------------------------------------------------------------
  相对 tp0509 的导出策略变更：
  1) 默认不再按 1° 格网导出大块特征栈；改为以样本点为中心的固定像元块（默认 512×512）
  2) 每块在「该点所在 UTM 带」下构造正方形窗口（边长 = CHIP_SIZE_PX * SCALE），导出时
     指定 crs + dimensions，保证像元对齐、尺寸严格为 CHIP_SIZE_PX
  3) 可选按「块间重叠面积占比」贪心去重（默认任意两保留块重叠 ≤60%）
  4) 分批 EXPORT_CHIP_POINT_OFFSET / EXPORT_CHIP_POINT_MAX，避免单次任务过多

  训练/验证的网格切分、RF、低置信掩膜等逻辑与 tp0509 一致。
*/
//从GEE copy过来的下载代码，现在已经下完了两千多个标注点的图像块，EXPORT_CHIP_POINT_OFFSET从0到两千多分批次下完了
var EXPORT_TAG = 'tp0518';

// ============================ 0) 参数配置 ============================
var YEAR = 2020;
var START_EARLY = YEAR + '-05-15';
var END_EARLY = YEAR + '-07-01';
var START_LATE = YEAR + '-07-01';
var END_LATE = YEAR + '-09-30';

var GRID_SIZE_DEG = 1;
var CLOUD_MAX = 40;
var SCALE = 10;
var TRAIN_GRID_RATIO = 0.7;
var RF_TREES = 220;
var RANDOM_SEED = 42;

var USE_S1 = true;
var MIN_S2_COUNT = 1;
var MIN_S1_COUNT = 0;

var EXPORT_TO_DRIVE = true;
var EXPORT_FOLDER = 'GEE_QTP_Wetland';

// ---------- 传统大块 / 格网导出（默认全关；深度学习 chip 请用下方点中心导出）----------
var EXPORT_CLASS_MOSAIC = false;
var EXPORT_LOWCONF_MASK = false;
var EXPORT_ACCURACY_BY_GRID = false;
var EXPORT_FEATURE_STACK = false;
var EXPORT_FEATURE_STACK_BY_GRID = false;
var FEATURE_STACK_GRID_OFFSET = 0;
var FEATURE_STACK_GRID_MAX = 30;
var EXPORT_GRID_CHIPS = false;
var MAX_EXPORT_GRIDS = 30;

// ---------- 点中心 512（可调）训练 chip 导出 ----------
// 点池：'all' 使用全部样本点；'train' / 'valid' 仅导出对应网格内的点（用于你只想导训练池时）
var CHIP_POINT_SOURCE = 'all';
// 像元块边长（正方形）；与 SCALE 组合为地面边长 CHIP_SIZE_PX * SCALE（默认 5120 m）
var CHIP_SIZE_PX = 512;
// 重叠缓解：贪心筛选，使任意两保留 chip 的重叠面积 / 单块面积 ≤ CHIP_MAX_OVERLAP
var CHIP_APPLY_OVERLAP_DEDUP = true;
var CHIP_MAX_OVERLAP = 0.6;  // 例如 0.6 表示重复面积不超过单块面积的 60%
// 本批导出的点在排序后 FC 中的起始索引与最大个数（0-based offset；与 tp0509 格网 offset 用法类似）
var EXPORT_CHIP_POINT_OFFSET = 2188;
var EXPORT_CHIP_POINT_MAX = 300;
// 分项：特征栈（float）与 RF 分类标签（uint8）；可只开其一调试
var EXPORT_CHIP_FEATURE_STACK = true;
var EXPORT_CHIP_CLASS_LABEL = true;
// 本批点清单 CSV（chip 序号、经纬度、类别、gridId、UTM EPSG），便于 AutoDL 侧对齐文件名
var EXPORT_CHIP_MANIFEST_CSV = true;

var USE_RF_BAND_SUBSET = true;
var RF_DROP_BANDS = ['BSI_E', 'BSI_L', 'LSWI_E', 'LSWI_L', 'VVVH_E', 'VVVH_L'];

var CLASS_WETLAND = 0;
var CLASS_WATER = 1;
var CLASS_LAKE = 2;
var CLASS_NONWET = 3;

// ============================ 1) 输入数据 ============================
var roi = table.geometry();
Map.centerObject(roi, 5);

var wetlandPts = ee.FeatureCollection(marsh);
var waterPts = ee.FeatureCollection(river);
var lakePts = ee.FeatureCollection(lake);
var nonWetlandPts = ee.FeatureCollection(nowet);

function setLabel(fc, label) {
  return fc.map(function(f) { return f.set('landcover', label); });
}

var allSamples = setLabel(wetlandPts, CLASS_WETLAND)
  .merge(setLabel(waterPts, CLASS_WATER))
  .merge(setLabel(lakePts, CLASS_LAKE))
  .merge(setLabel(nonWetlandPts, CLASS_NONWET));

print('样本总数', allSamples.size());
print('类别分布', allSamples.aggregate_histogram('landcover'));

// ============================ 2) 1°网格 ============================
function createGrid(geometry, sizeDeg) {
  var bounds = geometry.bounds().coordinates().get(0);
  var xs = ee.List(bounds).map(function(p) { return ee.List(p).get(0); });
  var ys = ee.List(bounds).map(function(p) { return ee.List(p).get(1); });

  var minX = ee.Number(xs.reduce(ee.Reducer.min())).floor();
  var maxX = ee.Number(xs.reduce(ee.Reducer.max())).ceil();
  var minY = ee.Number(ys.reduce(ee.Reducer.min())).floor();
  var maxY = ee.Number(ys.reduce(ee.Reducer.max())).ceil();

  var cols = ee.Number(maxX.subtract(minX).divide(sizeDeg)).ceil().int();
  var rows = ee.Number(maxY.subtract(minY).divide(sizeDeg)).ceil().int();
  var ids = ee.List.sequence(0, cols.multiply(rows).subtract(1));

  return ee.FeatureCollection(ids.map(function(i) {
    i = ee.Number(i);
    var col = i.mod(cols);
    var row = i.divide(cols).floor();
    var x0 = minX.add(col.multiply(sizeDeg));
    var y0 = minY.add(row.multiply(sizeDeg));
    var cell = ee.Geometry.Rectangle([x0, y0, x0.add(sizeDeg), y0.add(sizeDeg)], null, false);
    return ee.Feature(cell, {
      gridId: ee.String('G').cat(col.format('%02d')).cat('_').cat(row.format('%02d')),
      col: col,
      row: row
    });
  })).filterBounds(geometry);
}

var grids = createGrid(roi, GRID_SIZE_DEG);
print('网格数量', grids.size());
Map.addLayer(grids.style({color: 'FFAA00', fillColor: '00000000', width: 1}), {}, 'grid_1deg', true);

function attachGridId(fc) {
  return fc.map(function(f) {
    var hit = grids.filterBounds(f.geometry()).first();
    var gid = ee.Algorithms.If(hit, ee.Feature(hit).get('gridId'), 'OUT');
    return f.set('gridId', gid);
  }).filter(ee.Filter.neq('gridId', 'OUT'));
}

allSamples = attachGridId(allSamples);
print('带网格ID样本总数', allSamples.size());
print('每网格样本数', allSamples.aggregate_histogram('gridId'));

// ============================ 3) S2/S1 双时相特征 ============================
function maskS2SR(img) {
  var scl = img.select('SCL');
  var bad = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));
  return img.updateMask(bad.not())
    .divide(10000)
    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
            ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);
}

function buildS2Base(startDate, endDate, suffix) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_MAX))
    .map(maskS2SR);

  var p50 = col.reduce(ee.Reducer.percentile([50]))
    .rename(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);
  var count = col.select('red').count().rename('S2CNT_' + suffix);
  return p50.addBands(count).clip(roi);
}

function addIndices(img, suffix) {
  var ndvi = img.normalizedDifference(['nir', 'red']).rename('NDVI_' + suffix);
  var ndwi = img.normalizedDifference(['green', 'nir']).rename('NDWI_' + suffix);
  var mndwi = img.normalizedDifference(['green', 'swir1']).rename('MNDWI_' + suffix);
  var lswi = img.normalizedDifference(['nir', 'swir1']).rename('LSWI_' + suffix);
  var evi = img.expression(
    '2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))',
    {NIR: img.select('nir'), RED: img.select('red'), BLUE: img.select('blue')}
  ).rename('EVI_' + suffix);
  var bsi = img.expression(
    '((SWIR1+RED)-(NIR+BLUE))/((SWIR1+RED)+(NIR+BLUE))',
    {SWIR1: img.select('swir1'), RED: img.select('red'), NIR: img.select('nir'), BLUE: img.select('blue')}
  ).rename('BSI_' + suffix);
  return img.addBands([ndvi, ndwi, mndwi, lswi, evi, bsi]);
}

function buildS1Composite(startDate, endDate, suffix) {
  var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);

  var s1 = s1Col.median().clip(roi);
  var ratio = s1.select('VV').divide(s1.select('VH')).rename('VVVH_' + suffix);
  var count = s1Col.select('VV').count().rename('S1CNT_' + suffix);
  return s1.rename(['VV_' + suffix, 'VH_' + suffix]).addBands(ratio).addBands(count);
}

var s2Season = buildS2Base(YEAR + '-05-15', YEAR + '-09-30', 'S');
var s2EarlyBase = buildS2Base(START_EARLY, END_EARLY, 'E');
var s2LateBase = buildS2Base(START_LATE, END_LATE, 'L');

var s2EarlyCore = s2EarlyBase.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
  .unmask(s2Season.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']));
var s2LateCore = s2LateBase.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
  .unmask(s2Season.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']));

var s2Early = addIndices(s2EarlyCore, 'E').addBands(s2EarlyBase.select('S2CNT_E'));
var s2Late = addIndices(s2LateCore, 'L').addBands(s2LateBase.select('S2CNT_L'));

var dem = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(roi).rename('DEM');
var slope = ee.Terrain.slope(dem).rename('SLOPE');

var featureImage = s2Early.addBands(s2Late).addBands(dem).addBands(slope);
if (USE_S1) {
  var s1Early = buildS1Composite(START_EARLY, END_EARLY, 'E');
  var s1Late = buildS1Composite(START_LATE, END_LATE, 'L');
  featureImage = featureImage.addBands(s1Early).addBands(s1Late);
}

if (USE_RF_BAND_SUBSET) {
  featureImage = featureImage.select(
    ee.List(featureImage.bandNames()).removeAll(RF_DROP_BANDS)
  );
}

// ============================ 4) 低置信区掩膜 ============================
var highConfMask = featureImage.select('S2CNT_E').gte(MIN_S2_COUNT)
  .and(featureImage.select('S2CNT_L').gte(MIN_S2_COUNT));
if (USE_S1) {
  highConfMask = highConfMask
    .and(featureImage.select('S1CNT_E').gte(MIN_S1_COUNT))
    .and(featureImage.select('S1CNT_L').gte(MIN_S1_COUNT));
}
var lowConfMask = highConfMask.not().rename('LOW_CONF');

print('特征波段名', featureImage.bandNames());
print('RF 特征波段数', featureImage.bandNames().size());
Map.addLayer(s2EarlyCore, {bands: ['red', 'green', 'blue'], min: 0, max: 0.3}, 'S2_RGB_Early', false);
Map.addLayer(s2LateCore, {bands: ['red', 'green', 'blue'], min: 0, max: 0.3}, 'S2_RGB_Late', false);
Map.addLayer(featureImage.select('S2CNT_E'), {min: 0, max: 30, palette: ['#d73027', '#fee08b', '#1a9850']}, 'S2CNT_E', false);
Map.addLayer(featureImage.select('S2CNT_L'), {min: 0, max: 30, palette: ['#d73027', '#fee08b', '#1a9850']}, 'S2CNT_L', false);
Map.addLayer(lowConfMask.selfMask(), {palette: ['#7f7f7f']}, 'LOW_CONF_MASK', false);

// ============================ 5) 网格级训练/验证切分 ============================
var gridSplit = grids.randomColumn('randGrid', RANDOM_SEED);
var trainGridFC = gridSplit.filter(ee.Filter.lt('randGrid', TRAIN_GRID_RATIO));
var validGridFC = gridSplit.filter(ee.Filter.gte('randGrid', TRAIN_GRID_RATIO));

var trainGridIds = ee.Dictionary(trainGridFC.aggregate_histogram('gridId')).keys();
var validGridIds = ee.Dictionary(validGridFC.aggregate_histogram('gridId')).keys();

var trainPts = allSamples.filter(ee.Filter.inList('gridId', trainGridIds));
var validPts = allSamples.filter(ee.Filter.inList('gridId', validGridIds));

print('训练网格数', trainGridFC.size());
print('验证网格数', validGridFC.size());
print('训练点数(网格切分)', trainPts.size());
print('验证点数(网格切分)', validPts.size());

var trainTableRaw = featureImage.sampleRegions({
  collection: trainPts,
  properties: ['landcover', 'gridId'],
  scale: SCALE,
  tileScale: 8
});
var validTableRaw = featureImage.sampleRegions({
  collection: validPts,
  properties: ['landcover', 'gridId'],
  scale: SCALE,
  tileScale: 8
});

print('训练样本表(过滤前)', trainTableRaw.size());
print('验证样本表(过滤前)', validTableRaw.size());

var trainTable = trainTableRaw.filter(ee.Filter.gte('S2CNT_E', MIN_S2_COUNT))
  .filter(ee.Filter.gte('S2CNT_L', MIN_S2_COUNT));
var validTable = validTableRaw.filter(ee.Filter.gte('S2CNT_E', MIN_S2_COUNT))
  .filter(ee.Filter.gte('S2CNT_L', MIN_S2_COUNT));

if (USE_S1) {
  trainTable = trainTable.filter(ee.Filter.gte('S1CNT_E', MIN_S1_COUNT))
    .filter(ee.Filter.gte('S1CNT_L', MIN_S1_COUNT));
  validTable = validTable.filter(ee.Filter.gte('S1CNT_E', MIN_S1_COUNT))
    .filter(ee.Filter.gte('S1CNT_L', MIN_S1_COUNT));
}

print('MIN_S2_COUNT', MIN_S2_COUNT);
print('MIN_S1_COUNT', MIN_S1_COUNT);
print('训练样本表(过滤后)', trainTable.size());
print('验证样本表(过滤后)', validTable.size());

// ============================ 6) 全局RF训练与评估 ============================
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: RF_TREES,
  seed: RANDOM_SEED
}).train({
  features: trainTable,
  classProperty: 'landcover',
  inputProperties: featureImage.bandNames()
});

var globalClass = featureImage.classify(rf).rename('landcover').clip(roi);
Map.addLayer(globalClass, {
  min: 0, max: 3,
  palette: ['#2ca25f', '#41b6c4', '#225ea8', '#d9d9d9']
}, 'Global_RF_Classification', true);

var validPred = validTable.classify(rf, 'pred');
var cmGlobal = validPred.errorMatrix('landcover', 'pred');
print('全域OA', cmGlobal.accuracy());
print('全域Kappa', cmGlobal.kappa());

var gridIds = ee.List(grids.aggregate_array('gridId')).sort();
var perGridMetrics = ee.FeatureCollection(gridIds.map(function(gid) {
  gid = ee.String(gid);
  var v = validPred.filter(ee.Filter.eq('gridId', gid));
  var n = v.size();
  var hasValid = n.gt(0);
  var cm = ee.ConfusionMatrix(ee.Algorithms.If(hasValid, v.errorMatrix('landcover', 'pred'), [[0]]));
  return ee.Feature(null, {
    gridId: gid,
    validCount: n,
    OA: ee.Algorithms.If(hasValid, cm.accuracy(), null),
    Kappa: ee.Algorithms.If(hasValid, cm.kappa(), null),
    hasValid: hasValid
  });
}));

// ============================ 6b) 点中心 chip 几何（UTM）与重叠去重 ============================
var chipSideM = CHIP_SIZE_PX * SCALE;

// 去重用：经纬度 → 近似东/北向米偏移（避免贪心循环内反复 getInfo）
function lonLatToENMeters(lon, lat, lon0, lat0) {
  var R = 6378137;
  var latRad = lat * Math.PI / 180;
  var lat0Rad = lat0 * Math.PI / 180;
  var dLon = (lon - lon0) * Math.PI / 180;
  var dLat = (lat - lat0) * Math.PI / 180;
  var east = dLon * Math.cos(lat0Rad) * R;
  var north = dLat * R;
  return {east: east, north: north};
}

// 两轴对齐正方形 chip（边长 sideM、近似正北）的重叠面积 / 单块面积
function chipOverlapFractionEN(c1, c2, sideM) {
  var ox = Math.max(0, sideM - Math.abs(c2.east - c1.east));
  var oy = Math.max(0, sideM - Math.abs(c2.north - c1.north));
  return (ox * oy) / (sideM * sideM);
}

// 按纬度、经度排序后贪心保留：与已保留块重叠占比均 ≤ maxOverlap
function greedyDedupPointsByOverlap(features, maxOverlap, sideM) {
  var lonSum = 0;
  var latSum = 0;
  for (var j = 0; j < features.length; j++) {
    var c = features[j].geometry.coordinates;
    lonSum += c[0];
    latSum += c[1];
  }
  var lon0 = lonSum / features.length;
  var lat0 = latSum / features.length;

  var records = features.map(function(f) {
    var coords = f.geometry.coordinates;
    var lon = coords[0];
    var lat = coords[1];
    var en = lonLatToENMeters(lon, lat, lon0, lat0);
    return {
      lon: lon,
      lat: lat,
      east: en.east,
      north: en.north,
      properties: f.properties
    };
  });
  records.sort(function(a, b) {
    if (a.lat !== b.lat) {
      return a.lat < b.lat ? -1 : 1;
    }
    return a.lon < b.lon ? -1 : (a.lon > b.lon ? 1 : 0);
  });
  var kept = [];
  for (var i = 0; i < records.length; i++) {
    var cand = records[i];
    var accept = true;
    for (var k = 0; k < kept.length; k++) {
      if (chipOverlapFractionEN(cand, kept[k], sideM) > maxOverlap) {
        accept = false;
        break;
      }
    }
    if (accept) {
      kept.push(cand);
    }
  }
  return kept;
}

var chipPointPool = CHIP_POINT_SOURCE === 'train' ? trainPts
  : (CHIP_POINT_SOURCE === 'valid' ? validPts : allSamples);

var chipPoolCount = chipPointPool.size().getInfo();
var chipPoolFeatures = chipPointPool.getInfo().features;
var chipKeptRecords = CHIP_APPLY_OVERLAP_DEDUP
  ? greedyDedupPointsByOverlap(chipPoolFeatures, CHIP_MAX_OVERLAP, chipSideM)
  : chipPoolFeatures.map(function(f) {
      var coords = f.geometry.coordinates;
      return {lon: coords[0], lat: coords[1], properties: f.properties};
    });

var chipPointsPrepared = ee.FeatureCollection(chipKeptRecords.map(function(r) {
  return ee.Feature(ee.Geometry.Point([r.lon, r.lat]), r.properties);
}));

print('点中心 chip：点池来源', CHIP_POINT_SOURCE,
  '原始点数', chipPoolCount,
  '重叠去重', CHIP_APPLY_OVERLAP_DEDUP,
  '最大重叠占比', CHIP_MAX_OVERLAP,
  '去重后点数', chipKeptRecords.length);

function pointFeatureToChipFeature(f) {
  var pt = f.geometry();
  var lon = ee.Number(ee.List(pt.coordinates()).get(0));
  var lat = ee.Number(ee.List(pt.coordinates()).get(1));
  var zone = lon.add(180).divide(6).floor().add(1);
  var epsgNum = ee.Algorithms.If(
    lat.gte(0),
    ee.Number(32600).add(zone),
    ee.Number(32700).add(zone)
  );
  var crsStr = ee.String('EPSG:').cat(ee.Number(epsgNum).format('%d'));
  var crsProj = ee.Projection(crsStr);
  var xy = pt.transform(crsProj, ee.ErrorMargin(1));
  var x = ee.Number(ee.List(xy.coordinates()).get(0));
  var y = ee.Number(ee.List(xy.coordinates()).get(1));
  var halfM = ee.Number(SCALE).multiply(CHIP_SIZE_PX).divide(2);
  var rect = ee.Geometry.Rectangle(
    [x.subtract(halfM), y.subtract(halfM), x.add(halfM), y.add(halfM)],
    crsStr,
    false
  );
  return ee.Feature(rect, {
    landcover: f.get('landcover'),
    gridId: f.get('gridId'),
    chip_crs: crsStr
  });
}

var chipExportFC = chipPointsPrepared.map(pointFeatureToChipFeature);

// ============================ 7) 导出 ============================
if (EXPORT_TO_DRIVE) {
  var exportChips = EXPORT_CHIP_FEATURE_STACK || EXPORT_CHIP_CLASS_LABEL || EXPORT_CHIP_MANIFEST_CSV;
  var anyExport = EXPORT_CLASS_MOSAIC || EXPORT_LOWCONF_MASK || EXPORT_ACCURACY_BY_GRID ||
    EXPORT_FEATURE_STACK || EXPORT_FEATURE_STACK_BY_GRID || EXPORT_GRID_CHIPS || exportChips;

  if (!anyExport) {
    print('EXPORT_TO_DRIVE=true 但未勾选任何 EXPORT_* 分项，未创建任务');
  }

  if (EXPORT_CLASS_MOSAIC) {
    Export.image.toDrive({
      image: globalClass.toUint8(),
      description: 'QTP_Wetland_' + YEAR + '_mosaic_' + EXPORT_TAG,
      folder: EXPORT_FOLDER,
      fileNamePrefix: 'QTP_Wetland_' + YEAR + '_mosaic_' + EXPORT_TAG,
      region: roi,
      scale: SCALE,
      maxPixels: 1e13
    });
  }

  if (EXPORT_LOWCONF_MASK) {
    Export.image.toDrive({
      image: lowConfMask.toUint8(),
      description: 'QTP_Wetland_' + YEAR + '_lowconf_mask_' + EXPORT_TAG,
      folder: EXPORT_FOLDER,
      fileNamePrefix: 'QTP_Wetland_' + YEAR + '_lowconf_mask_' + EXPORT_TAG,
      region: roi,
      scale: SCALE,
      maxPixels: 1e13
    });
  }

  if (EXPORT_ACCURACY_BY_GRID) {
    Export.table.toDrive({
      collection: perGridMetrics,
      description: 'QTP_Wetland_' + YEAR + '_accuracy_by_grid_' + EXPORT_TAG,
      folder: EXPORT_FOLDER,
      fileNamePrefix: 'QTP_Wetland_' + YEAR + '_accuracy_by_grid_' + EXPORT_TAG,
      fileFormat: 'CSV'
    });
  }

  if (EXPORT_FEATURE_STACK && !EXPORT_FEATURE_STACK_BY_GRID) {
    Export.image.toDrive({
      image: featureImage.toFloat(),
      description: 'QTP_Wetland_' + YEAR + '_feature_stack_' + EXPORT_TAG,
      folder: EXPORT_FOLDER,
      fileNamePrefix: 'QTP_Wetland_' + YEAR + '_feature_stack_' + EXPORT_TAG,
      region: roi,
      scale: SCALE,
      maxPixels: 1e13
    });
  }

  if (EXPORT_FEATURE_STACK_BY_GRID) {
    var gridListFs = grids.toList(grids.size());
    var totalGridsFs = grids.size().getInfo();
    var fsStart = FEATURE_STACK_GRID_OFFSET;
    var fsEnd = Math.min(fsStart + FEATURE_STACK_GRID_MAX, totalGridsFs);
    if (fsStart >= totalGridsFs) {
      print('FEATURE_STACK_GRID_OFFSET 已 >= 网格总数，跳过特征栈格网导出');
    } else {
      print('特征栈格网导出批次：序号', fsStart + 1, '—', fsEnd, '/', totalGridsFs);
      for (var fi = fsStart; fi < fsEnd; fi++) {
        var gff = ee.Feature(gridListFs.get(fi));
        var gidFs = gff.get('gridId').getInfo();
        var geomFs = gff.geometry();
        var seqFs = fi + 1;
        var seqStrFs = ('0000' + seqFs).slice(-4);
        Export.image.toDrive({
          image: featureImage.clip(geomFs).toFloat(),
          description: 'QTP_Wetland_' + YEAR + '_fs_' + seqStrFs + '_' + gidFs + '_' + EXPORT_TAG,
          folder: EXPORT_FOLDER,
          fileNamePrefix: 'QTP_Wetland_' + YEAR + '_fs_' + seqStrFs + '_' + gidFs + '_' + EXPORT_TAG,
          region: geomFs,
          scale: SCALE,
          maxPixels: 1e13
        });
      }
    }
  }

  if (EXPORT_GRID_CHIPS) {
    var gridList = grids.toList(grids.size());
    var exportCount = ee.Number(grids.size()).min(MAX_EXPORT_GRIDS).getInfo();
    for (var gi = 0; gi < exportCount; gi++) {
      var gf = ee.Feature(gridList.get(gi));
      var gid = gf.get('gridId').getInfo();
      var geom = gf.geometry();
      Export.image.toDrive({
        image: globalClass.clip(geom).toUint8(),
        description: 'QTP_Wetland_' + YEAR + '_grid_' + gid + '_' + EXPORT_TAG,
        folder: EXPORT_FOLDER,
        fileNamePrefix: 'QTP_Wetland_' + YEAR + '_grid_' + gid + '_' + EXPORT_TAG,
        region: geom,
        scale: SCALE,
        maxPixels: 1e13
      });
    }
  }

  if (exportChips) {
    var chipTotal = chipExportFC.size().getInfo();
    var chipStart = EXPORT_CHIP_POINT_OFFSET;
    if (chipStart >= chipTotal) {
      print('EXPORT_CHIP_POINT_OFFSET 已 >= 点中心 chip 总数，跳过 chip 导出');
    } else {
      var chipEnd = Math.min(chipStart + EXPORT_CHIP_POINT_MAX, chipTotal);
      var chipBatchLen = chipEnd - chipStart;
      print('点中心 chip 导出批次：全局序号', chipStart + 1, '—', chipEnd, '/', chipTotal,
        '每点', CHIP_SIZE_PX, 'x', CHIP_SIZE_PX, '@', SCALE, 'm');

      var chipList = chipExportFC.toList(chipBatchLen, chipStart);

      if (EXPORT_CHIP_MANIFEST_CSV) {
        var manifestRows = ee.List.sequence(chipStart, chipEnd - 1).map(function(globalIdx) {
          globalIdx = ee.Number(globalIdx);
          var localIdx = globalIdx.subtract(chipStart);
          var feat = ee.Feature(chipList.get(localIdx));
          var centroid = feat.geometry().centroid(ee.ErrorMargin(1));
          var lon = ee.Number(ee.List(centroid.coordinates()).get(0));
          var lat = ee.Number(ee.List(centroid.coordinates()).get(1));
          return ee.Feature(null, {
            chip_index: globalIdx.add(1),
            lon: lon,
            lat: lat,
            landcover: feat.get('landcover'),
            gridId: feat.get('gridId'),
            chip_crs: feat.get('chip_crs'),
            file_suffix: ee.String('chip_').cat(globalIdx.add(1).format('%05d'))
          });
        });
        Export.table.toDrive({
          collection: ee.FeatureCollection(manifestRows),
          description: 'QTP_Wetland_' + YEAR + '_chip_manifest_' + chipStart + '_' + chipEnd + '_' + EXPORT_TAG,
          folder: EXPORT_FOLDER,
          fileNamePrefix: 'QTP_Wetland_' + YEAR + '_chip_manifest_' + chipStart + '_' + chipEnd + '_' + EXPORT_TAG,
          fileFormat: 'CSV'
        });
      }

      for (var ci = 0; ci < chipBatchLen; ci++) {
        var chipFeat = ee.Feature(chipList.get(ci));
        var chipRegion = chipFeat.geometry();
        var chipProps = chipFeat.toDictionary(['landcover', 'gridId', 'chip_crs']).getInfo();
        var crsExport = chipProps.chip_crs;
        var lc = chipProps.landcover;
        var gidStr = String(chipProps.gridId).replace(/[^A-Za-z0-9_-]/g, '_');
        var globalIdx1 = chipStart + ci + 1;
        var seqChip = ('00000' + globalIdx1).slice(-5);
        var prefix = 'QTP_Wetland_' + YEAR + '_chip_' + seqChip + '_lc' + lc + '_' + gidStr + '_' + EXPORT_TAG;
        var dimStr = String(CHIP_SIZE_PX) + 'x' + String(CHIP_SIZE_PX);

        if (EXPORT_CHIP_FEATURE_STACK) {
          Export.image.toDrive({
            image: featureImage.clip(chipRegion).toFloat(),
            description: 'fs_' + seqChip + '_' + EXPORT_TAG,
            folder: EXPORT_FOLDER,
            fileNamePrefix: prefix + '_features',
            region: chipRegion,
            crs: crsExport,
            dimensions: dimStr,
            maxPixels: 1e13
          });
        }
        if (EXPORT_CHIP_CLASS_LABEL) {
          Export.image.toDrive({
            image: globalClass.clip(chipRegion).toUint8(),
            description: 'lbl_' + seqChip + '_' + EXPORT_TAG,
            folder: EXPORT_FOLDER,
            fileNamePrefix: prefix + '_label',
            region: chipRegion,
            crs: crsExport,
            dimensions: dimStr,
            maxPixels: 1e13
          });
        }
      }
    }
  }
}

print('================ 运行建议 ================');
print('1) 点中心 chip：设 EXPORT_TO_DRIVE=true，再开 EXPORT_CHIP_FEATURE_STACK / EXPORT_CHIP_CLASS_LABEL');
print('2) 用 EXPORT_CHIP_POINT_OFFSET + EXPORT_CHIP_POINT_MAX 分批导出');
print('   CHIP_APPLY_OVERLAP_DEDUP + CHIP_MAX_OVERLAP 控制块间重复面积（默认≤60%）');
print('3) CHIP_POINT_SOURCE=all 导出全部点；train/valid 仅导出对应网格内点（与 RF 划分一致）');
print('4) 每块文件名含 chip 五位序号与 lc*、gridId；同序号 *_features 与 *_label 配对使用');
print('5) 仍需要大块/格网导出时可打开 EXPORT_FEATURE_STACK_BY_GRID 等（与 chip 独立）');
print('========================================');
