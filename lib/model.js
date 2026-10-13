const fs = require('fs'), path = require('path');
const M = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'model', 'model.json'), 'utf8'));
const REF_YEAR = M.ref_year;
const NOW_YEAR = new Date().getFullYear();

function evalTree(node, x) {
  while (node.leaf_value === undefined) {
    const v = x[node.split_feature];
    node = v <= node.threshold ? node.left_child : node.right_child;
  }
  return node.leaf_value;
}
function boost(trees, x) { let s = 0; for (const t of trees) s += evalTree(t.tree_structure, x); return s; }
function raw(cat, x) { const c = M.categories[cat]; return { p10: boost(c.trees['0.1'], x), p50: boost(c.trees['0.5'], x), p90: boost(c.trees['0.9'], x) }; }

function entry(cat, make, model) { return M.categories[cat].catalog.find(e => e.make === make && e.model === model); }
function features(cat, inp, e) {
  const c = M.categories[cat];
  const age = Math.max(0, inp.age), km = inp.km;
  const base = {
    te_make: c.enc.make[e.make] ?? c.prior, te_model: c.enc.model[e.model] ?? c.prior,
    age, km, lkm: Math.log1p(km), kmpa: km / Math.max(age, 0.5), owner_n: inp.owner, indiv: inp.indiv ? 1 : 0,
  };
  if (cat === 'car') {
    Object.assign(base, { trans: inp.trans ? 1 : 0, engine_cc: inp.engine_cc ?? e.engine_cc, power: inp.power ?? e.power, seats: inp.seats ?? e.seats,
      fuel_d: inp.fuel === 'Diesel' ? 1 : 0, fuel_c: inp.fuel === 'CNG' ? 1 : 0 });
  }
  return c.feature_names.map(n => base[n]);
}
function range(cat, x) {
  const r = raw(cat, x), w = M.categories[cat].metrics.widen_log;
  const lo = Math.min(r.p10, r.p50) - w, hi = Math.max(r.p90, r.p50) + w;
  return { low: Math.exp(lo), mid: Math.exp(r.p50), high: Math.exp(hi) };
}
const round = (n, step) => Math.round(n / step) * step;
const step = v => v >= 200000 ? 1000 : v >= 50000 ? 500 : 100;

class InputError extends Error {}

function validate(body) {
  const cat = body.category;
  if (!M.categories[cat]) throw new InputError('Category not supported yet. PriceLens only prices cars and bikes, because those are the only categories with verified data.');
  const e = entry(cat, body.make, body.model);
  if (!e) throw new InputError('That model is not in the verified data, so PriceLens will not guess a price for it.');
  const year = Math.round(Number(body.year)); const km = Number(body.km); const owner = Math.round(Number(body.owner || 1));
  if (year > REF_YEAR) throw new InputError(`The verified data only covers vehicles up to model-year ${REF_YEAR}. A ${year} vehicle is newer than anything in it, so PriceLens will not guess.`);
  if (!(year >= 1990)) throw new InputError('Year must be 1990 or later.');
  if (!(km >= 0 && km <= 1000000)) throw new InputError('Kilometres driven must be between 0 and 10,00,000.');
  if (!(owner >= 1 && owner <= 4)) throw new InputError('Owner count must be 1 to 4.');
  const out = { category: cat, make: e.make, model: e.model, year, km, owner, indiv: body.seller !== 'Dealer', age: REF_YEAR - year };
  if (cat === 'car') {
    const v = body.variant && e.variants.find(x => x.name === body.variant);
    out.fuel = ['Petrol', 'Diesel', 'CNG'].includes(body.fuel) ? body.fuel : (v ? v.fuel : (e.fuels[0] || 'Petrol'));
    out.trans = body.transmission ? body.transmission === 'Automatic' : (v ? !!v.trans : false);
    if (v) { out.variant = v.name; out.engine_cc = v.engine_cc; out.power = v.power; out.seats = v.seats; }
    else { const bf = e.by_fuel[out.fuel] || Object.values(e.by_fuel)[0]; out.engine_cc = bf.engine_cc; out.power = bf.power; out.seats = bf.seats; }
    if (!e.fuels.includes(out.fuel)) out.warnings = [`${e.model} has no ${out.fuel} listings in the data.`];
  }
  out.warnings = out.warnings || [];
  if (year < e.min_year - 3 || (year > e.max_year + 6)) out.warnings.push(`Listings for this model in the data cover ${e.min_year}-${e.max_year}; ${year} is outside that, so the estimate is less reliable.`);
  if (km > M.categories[cat].km_max) out.warnings.push('Kilometres are higher than anything in the data.');
  if (out.age > 25) out.warnings.push('Very old vehicle: few listings of this age exist in the data.');
  return { inp: out, e };
}

function appraise(body) {
  const { inp, e } = validate(body);
  const cat = inp.category, c = M.categories[cat];
  const x = features(cat, inp, e), r = range(cat, x);
  const s = step(r.mid);
  const result = { low: round(r.low, s), fair: round(r.mid, s), high: round(r.high, s) };
  // Quick/fair/patient labels = 10-90% of listing prices
  // Factors (sensitivity vs a reference vehicle: typical km for age, first owner)
  const typicalKm = Math.round(c.median_kmpa * Math.max(inp.age, 0.5));
  const refInp = { ...inp, km: typicalKm, owner: 1 };
  const refMid = range(cat, features(cat, refInp, e)).mid;
  const factors = [{ label: `Typical ${e.model} at ${inp.age} yr (${typicalKm.toLocaleString('en-IN')} km, 1st owner)`, amount: round(refMid, s), kind: 'base' }];
  const dKm = range(cat, features(cat, { ...refInp, km: inp.km }, e)).mid - refMid;
  if (Math.abs(dKm) >= s) factors.push({ label: `Kilometres: ${inp.km.toLocaleString('en-IN')} vs typical ${typicalKm.toLocaleString('en-IN')}`, amount: round(dKm, s) });
  const dOwn = range(cat, features(cat, { ...refInp, km: inp.km, owner: inp.owner }, e)).mid - range(cat, features(cat, { ...refInp, km: inp.km }, e)).mid;
  if (inp.owner > 1) factors.push({ label: `Owner count: ${inp.owner}`, amount: round(dOwn, s) });
  const dSel = range(cat, features(cat, inp, e)).mid - range(cat, features(cat, { ...inp, indiv: !inp.indiv }, e)).mid;
  if (Math.abs(dSel) >= s) factors.push({ label: inp.indiv ? 'Listed by individual (vs dealer)' : 'Listed by dealer (vs individual)', amount: round(dSel, s) });
  // Curve: same vehicle over the next 5 years
  const rate = inp.age >= 1 ? Math.min(25000, Math.max(2000, inp.km / inp.age)) : c.median_kmpa;
  const curve = [];
  for (let t = 0; t <= 5; t++) {
    const cr = range(cat, features(cat, { ...inp, age: inp.age + t, km: Math.round(inp.km + rate * t) }, e));
    curve.push({ t, age: inp.age + t, year: REF_YEAR - inp.age + inp.age + t, km: Math.round(inp.km + rate * t), low: round(cr.low, s), mid: round(cr.mid, s), high: round(cr.high, s) });
  }
  const m = c.metrics;
  const comps = comparables(cat, inp, e, 6);
  const waiting12 = curve[1].mid - curve[0].mid;
  return {
    input: inp, vintage: { year: REF_YEAR, label: `${REF_YEAR}-era listing prices` },
    basis: 'Asking prices on used-vehicle listings (not completed sales)',
    price: result, factors, curve, comps, rateKmPerYear: Math.round(rate), waiting12,
    dataset: { rows_model: e.n, rows_category: c.rows, median_price_model: Math.round(e.median_price) },
    quality: { median_abs_pct_error: +m.median_abs_pct_error.toFixed(1), range_coverage_pct: +m.range_coverage_calibrated_pct.toFixed(0), holdout_n: m.holdout_n },
    warnings: inp.warnings,
  };
}

function comparables(cat, inp, e, k) {
  const rows = M.categories[cat].comps.filter(r => r.make === e.make && r.model === e.model);
  const scored = rows.map(r => ({ r, d: Math.abs(r.year - inp.year) * 1.0 + Math.abs(r.km - inp.km) / 25000 + (cat === 'car' && r.fuel !== inp.fuel ? 1.5 : 0) + (cat === 'car' && !!r.trans !== !!inp.trans ? 1.5 : 0) + Math.abs(r.owner_n - inp.owner) * 0.3 }));
  scored.sort((a, b) => a.d - b.d);
  return scored.slice(0, k).map(({ r }) => ({ name: r.variant, year: r.year, km: r.km, owner: r.owner_n, seller: r.indiv ? 'Individual' : 'Dealer', fuel: r.fuel, trans: r.trans === undefined ? undefined : (r.trans ? 'Automatic' : 'Manual'), price: r.price }));
}

function dealCheck(body) {
  const a = appraise(body); const ask = Number(body.asking);
  if (!(ask > 0)) throw new InputError('Enter the seller\'s asking price.');
  const { low, fair, high } = a.price; const diff = ask - fair; const pct = diff / fair * 100;
  let verdict, tone;
  if (ask < low * 0.75) { verdict = 'Far below the usual range'; tone = 'warn'; }
  else if (ask < low) { verdict = 'Below the usual range'; tone = 'good'; }
  else if (ask <= high) { verdict = ask <= fair ? 'Within the range, at or under fair' : 'Within the range, above fair'; tone = 'ok'; }
  else { verdict = 'Above the usual range'; tone = 'bad'; }
  return { appraisal: a, asking: ask, diff: Math.round(diff), pct: +pct.toFixed(1), verdict, tone,
    counter: round(Math.max(low, Math.min(ask, fair * 0.97)), step(fair)), walkAway: round(high, step(fair)) };
}

function meta() {
  const cats = {};
  for (const [k, c] of Object.entries(M.categories)) {
    const makes = {};
    for (const e of c.catalog) (makes[e.make] ||= []).push({ model: e.model, n: e.n, min_year: e.min_year, max_year: e.max_year, median_price: Math.round(e.median_price),
      fuels: e.fuels, auto: e.auto, manual: e.manual, variants: e.variants, by_fuel: e.by_fuel });
    cats[k] = { makes, rows: c.rows, metrics: c.metrics, year_min: c.year_min, year_max: c.year_max, median_kmpa: Math.round(c.median_kmpa) };
  }
  return { ref_year: REF_YEAR, now_year: NOW_YEAR, categories: cats };
}

function market(cat) {
  const c = M.categories[cat]; if (!c) throw new InputError('Unknown category');
  const base = c.by_age.filter(r => r.age <= 3).reduce((s, r) => s + r.median_price * r.n, 0) / c.by_age.filter(r => r.age <= 3).reduce((s, r) => s + r.n, 0);
  const by_age = c.by_age.filter(r => r.age <= 20).map(r => ({ age: r.age, n: r.n, median_price: Math.round(r.median_price), median_km: Math.round(r.median_km), retention_pct: +(r.median_price / base * 100).toFixed(0) }));
  const top = [...c.catalog].sort((a, b) => b.n - a.n).slice(0, 12).map(e => ({ make: e.make, model: e.model, n: e.n, median_price: Math.round(e.median_price), median_km: Math.round(e.median_km), years: `${e.min_year}-${e.max_year}` }));
  return { category: cat, rows: c.rows, by_age, top, base_age_note: 'Retention = median asking price at each age vs the median at ages 0-3 in the same dataset.' };
}

const UNSUPPORTED = ['iphone', 'phone', 'mobile', 'samsung', 'oneplus', 'redmi', 'laptop', 'macbook', 'dell', 'hp ', 'lenovo', 'tv', 'television', 'fridge', 'refrigerator', 'washing', 'sofa', 'furniture', 'camera', 'ps5', 'playstation', 'xbox', 'watch', 'tablet', 'ipad', 'cycle', 'bicycle', 'truck', 'tractor', 'house', 'flat'];
function parseText(text) {
  const t = ' ' + String(text || '').toLowerCase().replace(/[,]/g, '') + ' ';
  let best = null, bestScore = 0;
  for (const [cat, c] of Object.entries(M.categories)) for (const e of c.catalog) {
    const toks = e.model.toLowerCase().split(/\s+/); let sc = 0;
    if (t.includes(' ' + e.make.toLowerCase() + ' ') || t.includes(e.make.toLowerCase())) sc += 2;
    const modelStr = e.model.toLowerCase();
    if (t.includes(modelStr)) sc += 3 + toks.length; else { const last = toks.slice(1).join(' '); if (last && t.includes(' ' + last)) sc += 2 + last.length / 10; }
    if (sc > bestScore) { bestScore = sc; best = { cat, e }; }
  }
  const hit = UNSUPPORTED.find(w => t.includes(w + ' ') || t.includes(' ' + w));
  if (!best || bestScore < 4) {
    return { ok: false, reason: hit ? `PriceLens does not price "${hit.trim()}" items yet. There is no verified resale data for that category, so it will not invent a number.` : 'I could not match that to a car or bike model in the verified data. Try something like "Maruti Swift 2017 petrol 45000 km".', supported: ['Cars (CarDekho listings)', 'Bikes and scooters (BikeDekho listings)'] };
  }
  const year = (t.match(/\b(19[89]\d|20[0-2]\d)\b/) || [])[1];
  let km = (t.match(/(\d+(?:\.\d+)?)\s?k\s?(?:km|kms)?\b/) || [])[1]; if (km) km = Math.round(parseFloat(km) * 1000);
  else km = (t.match(/(\d{3,7})\s?(?:km|kms|kilomet)/) || [])[1];
  const own = /\b(fourth|4th)\b/.test(t) ? 4 : /\b(third|3rd)\b/.test(t) ? 3 : /\b(second|2nd)\b/.test(t) ? 2 : /\b(first|1st|single|one)\b/.test(t) ? 1 : undefined;
  const fuel = /diesel/.test(t) ? 'Diesel' : /cng/.test(t) ? 'CNG' : /petrol/.test(t) ? 'Petrol' : undefined;
  const trans = /automatic|\bauto\b|amt|cvt/.test(t) ? 'Automatic' : /manual/.test(t) ? 'Manual' : undefined;
  return { ok: true, category: best.cat, make: best.e.make, model: best.e.model, year: year && +year, km: km && +km, owner: own, fuel, transmission: trans };
}
module.exports = { appraise, dealCheck, meta, market, parseText, InputError, M, evalTree, boost, raw };
