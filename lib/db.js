// Postgres in production (DATABASE_URL). Without it, an embedded Postgres (PGlite) stored in ./.pgdata is used for local development only.
let q;
async function init() {
  if (process.env.DATABASE_URL) {
    const { Pool } = require('pg');
    const pool = new Pool({ connectionString: process.env.DATABASE_URL, ssl: /localhost|127\.0\.0\.1/.test(process.env.DATABASE_URL) ? false : { rejectUnauthorized: false } });
    q = async (sql, p = []) => (await pool.query(sql, p)).rows;
    console.log('Database: Postgres via DATABASE_URL');
  } else {
    const { PGlite } = require('@electric-sql/pglite');
    const db = new PGlite(process.env.PGLITE_DIR || './.pgdata');
    await db.waitReady;
    q = async (sql, p = []) => (await db.query(sql, p)).rows;
    console.log('Database: embedded PGlite (development only). Set DATABASE_URL for production.');
  }
  await q(`CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL, pass_hash TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT now())`);
  await q(`CREATE TABLE IF NOT EXISTS appraisals (id TEXT PRIMARY KEY, user_id INT REFERENCES users(id) ON DELETE SET NULL, category TEXT NOT NULL, input JSONB NOT NULL, result JSONB NOT NULL, created_at TIMESTAMPTZ DEFAULT now())`);
  await q(`CREATE TABLE IF NOT EXISTS vault_items (id SERIAL PRIMARY KEY, user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE, label TEXT NOT NULL, category TEXT NOT NULL, input JSONB NOT NULL, low INT, fair INT, high INT, target INT, appraised_at TIMESTAMPTZ DEFAULT now(), created_at TIMESTAMPTZ DEFAULT now())`);
  await q(`CREATE INDEX IF NOT EXISTS appraisals_user ON appraisals(user_id, created_at DESC)`);
}
module.exports = { init, query: (...a) => q(...a) };
