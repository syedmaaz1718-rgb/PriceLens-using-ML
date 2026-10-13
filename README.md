# PriceLens

Used car and bike price estimator built on real listing data. Node + Express backend, Postgres database, LightGBM quantile models trained offline and evaluated in Node.

## What it does
- Estimates the typical **asking price** for a used car or bike with a low / fair / high range.
- Depreciation curve (5 years), "what moved the price", cost of waiting 12 months, closest real listings.
- Deal checker (seller's price vs the range) with a red-flag checklist.
- Vault (accounts, saved items, sell targets, re-appraise), history, market pulse, full data and method page.
- Refuses anything it has no verified data for (phones, laptops, vehicles newer than 2020, unknown models).

## Honest limits (also shown in the app on /methodology)
- Data is **asking prices**, around **2020**, not completed sales.
- Car data: CarDekho vehicle dataset (Kaggle, Nehal Birla, "Car details v3.csv"). Bike data: "Bike Prices.csv" from YBIFoundation/Dataset. Neither shows a licence. Fine for learning and portfolio use; check the terms before anything commercial.
- Accuracy numbers shown in the app are measured on held-out listings during training (see /methodology). Nothing is made up.

## Deploy on Render (steps)
1. Create a free Postgres database on neon.tech (copy the connection string).
2. Push this folder to a GitHub repo (never commit `.env`).
3. Render > New > Web Service > connect the repo. Runtime Node, build `npm install`, start `npm start`.
4. Environment: `DATABASE_URL` = your Neon string, `JWT_SECRET` = any long random text, `NODE_ENV` = production.
5. Deploy. Tables are created automatically on first start.

## Run locally
```
npm install
npm start        # uses an embedded Postgres in ./.pgdata when DATABASE_URL is not set
```
## Retrain the model
```
pip install pandas numpy scikit-learn lightgbm
npm run train    # rewrites model/model.json
```
To use fresher data, replace the CSVs in `data/` with the same columns and retrain.
