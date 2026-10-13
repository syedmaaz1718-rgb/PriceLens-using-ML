const express = require('express'), cookieParser = require('cookie-parser'), bcrypt = require('bcryptjs'), jwt = require('jsonwebtoken'), crypto = require('crypto'), path = require('path');
const db = require('./lib/db'); const model = require('./lib/model');
const app = express();
const SECRET = process.env.JWT_SECRET || (console.warn('JWT_SECRET not set: using a random secret (sessions reset on restart).'), crypto.randomBytes(32).toString('hex'));
app.use(express.json({ limit: '50kb' })); app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public'), { extensions: ['html'] }));

const wrap = fn => (req, res) => fn(req, res).catch(e => {
  if (e instanceof model.InputError) return res.status(400).json({ error: e.message });
  console.error(e); res.status(500).json({ error: 'Something went wrong on the server.' });
});
function auth(req, _res, next) { try { const t = req.cookies.pl; if (t) req.user = jwt.verify(t, SECRET); } catch { } next(); }
function need(req, res, next) { if (!req.user) return res.status(401).json({ error: 'Please sign in first.' }); next(); }
app.use(auth);
const setCookie = (res, u) => res.cookie('pl', jwt.sign({ id: u.id, name: u.name, email: u.email }, SECRET, { expiresIn: '30d' }), { httpOnly: true, sameSite: 'lax', secure: process.env.NODE_ENV === 'production', maxAge: 30 * 86400e3 });

app.get('/api/health', (_q, r) => r.json({ ok: true }));
app.get('/api/meta', (_q, r) => r.json(model.meta()));
app.get('/api/me', (q, r) => r.json({ user: q.user ? { id: q.user.id, name: q.user.name, email: q.user.email } : null }));
app.post('/api/auth/register', wrap(async (q, r) => {
  const { name, email, password } = q.body || {};
  if (!name || !/^\S+@\S+\.\S+$/.test(email || '') || (password || '').length < 8) return r.status(400).json({ error: 'Enter your name, a valid email and a password of at least 8 characters.' });
  const ex = await db.query('SELECT 1 FROM users WHERE email=$1', [email.toLowerCase()]);
  if (ex.length) return r.status(409).json({ error: 'That email already has an account. Sign in instead.' });
  const u = (await db.query('INSERT INTO users(email,name,pass_hash) VALUES($1,$2,$3) RETURNING id,name,email', [email.toLowerCase(), name.trim().slice(0, 60), await bcrypt.hash(password, 10)]))[0];
  setCookie(r, u); r.json({ user: u });
}));
app.post('/api/auth/login', wrap(async (q, r) => {
  const { email, password } = q.body || {};
  const u = (await db.query('SELECT * FROM users WHERE email=$1', [(email || '').toLowerCase()]))[0];
  if (!u || !(await bcrypt.compare(password || '', u.pass_hash))) return r.status(401).json({ error: 'Wrong email or password.' });
  setCookie(r, u); r.json({ user: { id: u.id, name: u.name, email: u.email } });
}));
app.post('/api/auth/logout', (_q, r) => { r.clearCookie('pl'); r.json({ ok: true }); });

app.post('/api/parse', wrap(async (q, r) => r.json(model.parseText((q.body || {}).text))));
app.post('/api/estimate', wrap(async (q, r) => { const a = model.appraise(q.body || {}); r.json({ price: a.price, warnings: a.warnings, input: a.input, dataset: a.dataset }); }));
app.post('/api/appraise', wrap(async (q, r) => {
  const result = model.appraise(q.body || {}); const id = crypto.randomBytes(5).toString('hex');
  await db.query('INSERT INTO appraisals(id,user_id,category,input,result) VALUES($1,$2,$3,$4,$5)', [id, q.user ? q.user.id : null, result.input.category, JSON.stringify(result.input), JSON.stringify(result)]);
  r.json({ id, result });
}));
app.get('/api/appraisals/:id', wrap(async (q, r) => {
  const a = (await db.query('SELECT id,result,created_at FROM appraisals WHERE id=$1', [q.params.id]))[0];
  if (!a) return r.status(404).json({ error: 'Appraisal not found.' }); r.json(a);
}));
app.post('/api/deal-check', wrap(async (q, r) => r.json(model.dealCheck(q.body || {}))));
app.get('/api/market/:cat', wrap(async (q, r) => r.json(model.market(q.params.cat))));
app.get('/api/history', need, wrap(async (q, r) => r.json(await db.query('SELECT id,category,input,(result->\'price\') AS price,created_at FROM appraisals WHERE user_id=$1 ORDER BY created_at DESC LIMIT 50', [q.user.id]))));

const itemLabel = i => `${i.year} ${i.model}`;
app.get('/api/vault', need, wrap(async (q, r) => {
  const items = await db.query('SELECT * FROM vault_items WHERE user_id=$1 ORDER BY created_at DESC', [q.user.id]);
  const out = items.map(it => { try { const a = model.appraise(it.input); return { ...it, curve: a.curve, waiting12: a.waiting12 }; } catch { return it; } });
  r.json({ items: out, total: items.reduce((s, i) => s + (i.fair || 0), 0), change12: out.reduce((s, i) => s + (i.waiting12 || 0), 0) });
}));
app.post('/api/vault', need, wrap(async (q, r) => {
  const a = model.appraise(q.body.input || q.body); const t = q.body.target ? Math.round(Number(q.body.target)) : null;
  const row = (await db.query('INSERT INTO vault_items(user_id,label,category,input,low,fair,high,target) VALUES($1,$2,$3,$4,$5,$6,$7,$8) RETURNING *', [q.user.id, q.body.label || itemLabel(a.input), a.input.category, JSON.stringify(a.input), a.price.low, a.price.fair, a.price.high, t]))[0];
  r.json(row);
}));
app.patch('/api/vault/:id', need, wrap(async (q, r) => {
  const t = q.body.target === null || q.body.target === '' ? null : Math.round(Number(q.body.target));
  r.json((await db.query('UPDATE vault_items SET target=$1 WHERE id=$2 AND user_id=$3 RETURNING *', [t, q.params.id, q.user.id]))[0] || {});
}));
app.post('/api/vault/:id/reappraise', need, wrap(async (q, r) => {
  const it = (await db.query('SELECT * FROM vault_items WHERE id=$1 AND user_id=$2', [q.params.id, q.user.id]))[0]; if (!it) return r.status(404).json({ error: 'Item not found.' });
  const a = model.appraise(it.input);
  r.json((await db.query('UPDATE vault_items SET low=$1,fair=$2,high=$3,appraised_at=now() WHERE id=$4 RETURNING *', [a.price.low, a.price.fair, a.price.high, it.id]))[0]);
}));
app.delete('/api/vault/:id', need, wrap(async (q, r) => { await db.query('DELETE FROM vault_items WHERE id=$1 AND user_id=$2', [q.params.id, q.user.id]); r.json({ ok: true }); }));

app.use('/api', (_q, r) => r.status(404).json({ error: 'Not found' }));
app.use((_q, r) => r.status(404).sendFile(path.join(__dirname, 'public', '404.html')));
const port = process.env.PORT || 3000;
db.init().then(() => app.listen(port, () => console.log('PriceLens on', port))).catch(e => { console.error('DB init failed', e); process.exit(1); });
