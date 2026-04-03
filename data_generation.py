"""
Synthetic Sales Data Generator for GCC Retail Forecasting
Generates a hyper-realistic multi-dimensional dataset with business realism.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# MASTER CONFIGURATION
# ─────────────────────────────────────────────────────────────

REGIONS_CITIES = {
    'UAE': ['Dubai', 'Abu Dhabi', 'Sharjah', 'Ajman'],
    'KSA': ['Riyadh', 'Jeddah', 'Dammam', 'Mecca'],
    'Oman': ['Muscat', 'Salalah', 'Sohar', 'Nizwa'],
    'Qatar': ['Doha', 'Al Wakrah', 'Al Khor', 'Lusail'],
}

CATEGORIES = {
    'Electronics': {
        'sub_categories': ['Smartphones', 'Laptops', 'Tablets', 'Headphones', 'Smart Watches', 'Cameras'],
        'price_range': (150, 3500),
        'base_demand': (1, 8),
        'frequency': 'low',
    },
    'Grocery': {
        'sub_categories': ['Dairy', 'Bakery', 'Beverages', 'Snacks', 'Frozen Foods', 'Fresh Produce'],
        'price_range': (2, 50),
        'base_demand': (20, 80),
        'frequency': 'high',
    },
    'Fashion': {
        'sub_categories': ['Men Clothing', 'Women Clothing', 'Kids Clothing', 'Footwear', 'Accessories', 'Sportswear'],
        'price_range': (20, 800),
        'base_demand': (3, 15),
        'frequency': 'medium',
    },
    'Home & Living': {
        'sub_categories': ['Furniture', 'Kitchen', 'Bedding', 'Lighting', 'Decor', 'Storage'],
        'price_range': (15, 1200),
        'base_demand': (2, 10),
        'frequency': 'low',
    },
    'Beauty': {
        'sub_categories': ['Skincare', 'Fragrance', 'Haircare', 'Makeup', 'Personal Care', 'Wellness'],
        'price_range': (10, 400),
        'base_demand': (5, 20),
        'frequency': 'medium',
    },
    'Sports': {
        'sub_categories': ['Fitness Equipment', 'Sportswear', 'Outdoor Gear', 'Supplements', 'Footwear', 'Accessories'],
        'price_range': (15, 600),
        'base_demand': (2, 12),
        'frequency': 'low',
    },
}

PRODUCT_NAME_PREFIXES = {
    'Electronics': ['ProTech', 'SmartEdge', 'NovaTech', 'ZenDigital', 'VoltX', 'PixelMax'],
    'Grocery': ['FreshChoice', 'NaturePure', 'DailyDelight', 'OrganicBest', 'FarmFresh', 'GoldenHarvest'],
    'Fashion': ['UrbanStyle', 'ClassicFit', 'TrendWave', 'EleganceX', 'VogueWear', 'ModernEdge'],
    'Home & Living': ['HomePlus', 'CozyNest', 'LivingArt', 'EliteHome', 'NestCraft', 'UrbanSpace'],
    'Beauty': ['GlowUp', 'PureSkin', 'LuxeBeauty', 'VelvetCare', 'RadiantX', 'BloomEssence'],
    'Sports': ['ProFit', 'PowerMax', 'ActiveEdge', 'SportElite', 'PeakForce', 'FlexZone'],
}

# Ramadan approximate start dates (moves ~11 days earlier each year)
RAMADAN_STARTS = {
    2020: datetime(2020, 4, 24),
    2021: datetime(2021, 4, 13),
    2022: datetime(2022, 4, 2),
    2023: datetime(2023, 3, 23),
    2024: datetime(2024, 3, 11),
}

NATIONAL_HOLIDAYS = {
    'UAE': [(12, 2), (12, 3)],    # National Day
    'KSA': [(9, 23)],             # National Day
    'Oman': [(11, 18)],           # National Day
    'Qatar': [(12, 18)],          # National Day
}


def _generate_product_master(num_products=500, seed=42):
    """Generate a product master table with realistic names."""
    rng = np.random.default_rng(seed)
    products = []
    cat_names = list(CATEGORIES.keys())
    # Weighted distribution: more Grocery, fewer Electronics
    cat_weights = np.array([0.12, 0.30, 0.18, 0.15, 0.15, 0.10])
    cat_weights /= cat_weights.sum()
    cat_assignments = rng.choice(cat_names, size=num_products, p=cat_weights)

    for i in range(num_products):
        cat = cat_assignments[i]
        cfg = CATEGORIES[cat]
        sub_cat = rng.choice(cfg['sub_categories'])
        prefix = rng.choice(PRODUCT_NAME_PREFIXES[cat])
        product_name = f"{prefix} {sub_cat.split()[0]} {rng.integers(100, 999)}"
        base_price = round(rng.uniform(*cfg['price_range']), 2)
        products.append({
            'product_id': f'PRD-{i+1:04d}',
            'product_name': product_name,
            'category': cat,
            'sub_category': sub_cat,
            'base_price': base_price,
            'base_demand_low': cfg['base_demand'][0],
            'base_demand_high': cfg['base_demand'][1],
        })
    return pd.DataFrame(products)


def _generate_store_master(num_stores=50, seed=42):
    """Generate a store master table distributed across GCC."""
    rng = np.random.default_rng(seed)
    stores = []
    regions = list(REGIONS_CITIES.keys())
    # UAE and KSA get more stores
    region_weights = np.array([0.35, 0.30, 0.20, 0.15])
    region_weights /= region_weights.sum()
    region_assignments = rng.choice(regions, size=num_stores, p=region_weights)

    for i in range(num_stores):
        region = region_assignments[i]
        city = rng.choice(REGIONS_CITIES[region])
        stores.append({
            'store_id': f'STR-{i+1:03d}',
            'region': region,
            'city': city,
        })
    return pd.DataFrame(stores)


def _build_calendar(start_date, end_date):
    """Build a rich calendar DataFrame with all temporal and holiday features."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    cal = pd.DataFrame({'date': dates})
    cal['day_of_week'] = cal['date'].dt.dayofweek
    cal['month'] = cal['date'].dt.month
    cal['year'] = cal['date'].dt.year
    cal['day_of_year'] = cal['date'].dt.dayofyear

    # Weekend flag (Friday=4, Saturday=5 in GCC)
    cal['weekend_flag'] = cal['day_of_week'].isin([4, 5]).astype(int)

    # Holiday flag — Ramadan (30 days), Eid Al-Fitr (3 days after Ramadan), Eid Al-Adha (~70 days after Ramadan start)
    cal['holiday_flag'] = 0
    cal['ramadan_flag'] = 0
    cal['eid_flag'] = 0

    for year, ram_start in RAMADAN_STARTS.items():
        ram_mask = (cal['date'] >= pd.Timestamp(ram_start)) & (cal['date'] < pd.Timestamp(ram_start) + pd.Timedelta(days=30))
        cal.loc[ram_mask, 'ramadan_flag'] = 1
        cal.loc[ram_mask, 'holiday_flag'] = 1

        eid_fitr_start = pd.Timestamp(ram_start) + pd.Timedelta(days=30)
        eid_fitr_mask = (cal['date'] >= eid_fitr_start) & (cal['date'] < eid_fitr_start + pd.Timedelta(days=3))
        cal.loc[eid_fitr_mask, 'eid_flag'] = 1
        cal.loc[eid_fitr_mask, 'holiday_flag'] = 1

        eid_adha_start = pd.Timestamp(ram_start) + pd.Timedelta(days=70)
        eid_adha_mask = (cal['date'] >= eid_adha_start) & (cal['date'] < eid_adha_start + pd.Timedelta(days=4))
        cal.loc[eid_adha_mask, 'eid_flag'] = 1
        cal.loc[eid_adha_mask, 'holiday_flag'] = 1

    # New Year
    for y in range(2020, 2026):
        ny_mask = cal['date'] == pd.Timestamp(datetime(y, 1, 1))
        cal.loc[ny_mask, 'holiday_flag'] = 1

    # Economic index: gradual inflation with a dip around COVID period (2020-2021)
    total_days = len(cal)
    base_econ = np.linspace(0.95, 1.15, total_days)
    # COVID dip in 2020
    covid_mask = (cal['date'] >= '2020-03-01') & (cal['date'] <= '2020-09-30')
    base_econ[covid_mask.values] *= 0.85
    # Recovery
    recovery_mask = (cal['date'] >= '2020-10-01') & (cal['date'] <= '2021-06-30')
    recovery_factor = np.linspace(0.90, 1.0, recovery_mask.sum())
    base_econ[recovery_mask.values] *= recovery_factor
    cal['economic_index'] = base_econ

    # Weather index (hot summers, mild winters) — GCC specific
    cal['weather_index'] = 25 + 15 * np.sin(2 * np.pi * (cal['day_of_year'].values - 80) / 365) + \
                           np.random.default_rng(0).normal(0, 2, total_days)
    cal['weather_index'] = cal['weather_index'].clip(15, 52)

    return cal


def generate_sales_data(num_stores=50, num_products=500, 
                        start_date='2020-01-01', end_date='2024-12-31',
                        products_per_store=50, seed=42):
    """
    Generate a hyper-realistic synthetic sales dataset for GCC retail.
    
    Uses a sampling strategy: each store carries a random subset of products.
    Total rows ≈ num_stores × products_per_store × num_days
    """
    rng = np.random.default_rng(seed)
    
    print("📦 Generating product master...")
    products_df = _generate_product_master(num_products, seed)
    
    print("🏬 Generating store master...")
    stores_df = _generate_store_master(num_stores, seed)
    
    print("📅 Building calendar...")
    cal = _build_calendar(start_date, end_date)
    num_days = len(cal)
    
    print(f"⚡ Generating sales data ({num_stores} stores × {products_per_store} products × {num_days} days)...")
    
    all_chunks = []
    total_combos = num_stores * products_per_store
    progress_step = max(1, total_combos // 10)
    combo_count = 0
    
    for _, store in stores_df.iterrows():
        store_id = store['store_id']
        region = store['region']
        city = store['city']
        
        # Each store carries a random subset of products
        store_products = products_df.sample(n=min(products_per_store, len(products_df)), 
                                           random_state=rng.integers(0, 100000))
        
        for _, product in store_products.iterrows():
            combo_count += 1
            if combo_count % progress_step == 0:
                pct = combo_count / total_combos * 100
                print(f"   ⏳ {pct:.0f}% complete ({combo_count}/{total_combos} store-product combos)...")
            
            pid = product['product_id']
            pname = product['product_name']
            cat = product['category']
            sub_cat = product['sub_category']
            base_price = product['base_price']
            bd_low = product['base_demand_low']
            bd_high = product['base_demand_high']
            
            # ── Time Series Generation (Vectorized) ──
            
            # 1. Base demand
            base_demand = rng.uniform(bd_low, bd_high, num_days)
            
            # 2. Growth trend (gradual upward with slight noise)
            trend = np.linspace(1.0, 1.35, num_days) + rng.normal(0, 0.01, num_days)
            
            # 3. Weekly seasonality (weekend spikes)
            weekend_factor = np.where(cal['weekend_flag'].values == 1, 
                                      rng.uniform(1.2, 1.5), 1.0)
            
            # 4. Monthly seasonality
            month_vals = cal['month'].values
            monthly_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (month_vals - 1) / 12)
            # Q4 boost for Electronics/Fashion
            if cat in ['Electronics', 'Fashion']:
                q4_mask = month_vals >= 10
                monthly_factor[q4_mask] *= 1.25
            
            # 5. Holiday / Ramadan / Eid factors
            holiday_factor = np.ones(num_days)
            # Ramadan: moderate boost (pre-Iftar shopping)
            holiday_factor[cal['ramadan_flag'].values == 1] *= rng.uniform(1.3, 1.6)
            # Eid: major spike
            holiday_factor[cal['eid_flag'].values == 1] *= rng.uniform(1.8, 2.5)
            # Generic holiday
            generic_holiday = (cal['holiday_flag'].values == 1) & \
                              (cal['ramadan_flag'].values == 0) & \
                              (cal['eid_flag'].values == 0)
            holiday_factor[generic_holiday] *= rng.uniform(1.2, 1.4)
            
            # National holiday boost for matching region
            if region in NATIONAL_HOLIDAYS:
                for m, d in NATIONAL_HOLIDAYS[region]:
                    nat_mask = (cal['month'].values == m) & (cal['date'].dt.day.values == d)
                    holiday_factor[nat_mask] *= 1.5
            
            # 6. Economic index impact
            econ_factor = cal['economic_index'].values
            
            # 7. Discount
            discount_pct = np.zeros(num_days)
            # ~25% of days have some discount
            disc_mask = rng.random(num_days) < 0.25
            discount_pct[disc_mask] = rng.choice([5, 10, 15, 20, 25, 30], size=disc_mask.sum())
            # Holiday periods get bigger discounts
            hol_disc_mask = cal['holiday_flag'].values == 1
            discount_pct[hol_disc_mask] = np.maximum(discount_pct[hol_disc_mask], 
                                                       rng.choice([15, 20, 25, 30, 35], 
                                                                  size=hol_disc_mask.sum()))
            discount_factor = 1.0 + (discount_pct / 100) * 0.8  # Discounts boost sales
            
            # 8. Marketing spend (sporadic, with lagged effect)
            marketing_spend = np.zeros(num_days)
            mkt_mask = rng.random(num_days) < 0.15  # 15% of days have marketing
            marketing_spend[mkt_mask] = rng.uniform(50, 500, mkt_mask.sum())
            # Lagged marketing effect (3-7 day lag)
            mkt_effect = np.ones(num_days)
            for lag in range(3, 8):
                shifted = np.roll(marketing_spend, lag)
                shifted[:lag] = 0
                mkt_effect += shifted / 2000  # Subtle lagged boost
            
            # 9. Stock availability
            stock_available = rng.integers(20, 150, num_days)
            # 5% chance of stockout
            stockout_mask = rng.random(num_days) < 0.05
            stock_available[stockout_mask] = 0
            # Supply chain delays can reduce stock
            supply_delay = np.zeros(num_days, dtype=int)
            delay_mask = rng.random(num_days) < 0.08
            supply_delay[delay_mask] = rng.integers(1, 7, delay_mask.sum())
            # Delays reduce next-day stock
            for i in range(1, num_days):
                if supply_delay[i-1] > 0:
                    stock_available[i] = max(0, stock_available[i] - rng.integers(10, 40))
            
            # 10. Competitor pricing
            competitor_price = base_price * rng.uniform(0.85, 1.15, num_days)
            price_competitiveness = np.clip(competitor_price / base_price, 0.8, 1.2)
            
            # 11. Customer footfall (store-level, correlated with weekends/holidays)
            footfall_base = rng.uniform(200, 800)
            customer_footfall = (footfall_base * weekend_factor * holiday_factor * 
                                 econ_factor * rng.uniform(0.8, 1.2, num_days)).astype(int)
            
            # ── Calculate Units Sold ──
            units_raw = (base_demand * trend * weekend_factor * monthly_factor * 
                        holiday_factor * econ_factor * discount_factor * mkt_effect * 
                        price_competitiveness)
            
            # Add noise
            units_raw += rng.normal(0, max(1, bd_low * 0.3), num_days)
            units_raw = np.maximum(0, units_raw).astype(int)
            
            # Stock constraint: can't sell more than available
            units_sold = np.minimum(units_raw, stock_available)
            # Zero stock = zero sales
            units_sold[stock_available == 0] = 0
            
            # Conversion rate = units_sold / customer_footfall (capped)
            conversion_rate = np.where(customer_footfall > 0, 
                                       np.clip(units_sold / customer_footfall, 0, 0.35),
                                       0.0)
            
            # Effective price after discount
            effective_price = base_price * (1 - discount_pct / 100)
            revenue = units_sold * effective_price
            
            # ── Build chunk DataFrame ──
            chunk = pd.DataFrame({
                'date': cal['date'].values,
                'store_id': store_id,
                'product_id': pid,
                'product_name': pname,
                'category': cat,
                'sub_category': sub_cat,
                'region': region,
                'city': city,
                'units_sold': units_sold,
                'revenue': np.round(revenue, 2),
                'price_per_unit': base_price,
                'discount_percentage': discount_pct,
                'marketing_spend': np.round(marketing_spend, 2),
                'customer_footfall': customer_footfall,
                'conversion_rate': np.round(conversion_rate, 4),
                'holiday_flag': cal['holiday_flag'].values,
                'weekend_flag': cal['weekend_flag'].values,
                'competitor_price': np.round(competitor_price, 2),
                'stock_available': stock_available,
                'supply_chain_delay_days': supply_delay,
                'weather_index': np.round(cal['weather_index'].values, 1),
                'economic_index': np.round(cal['economic_index'].values, 4),
            })
            all_chunks.append(chunk)
    
    print("🔗 Concatenating all data...")
    df = pd.concat(all_chunks, ignore_index=True)
    df = df.sort_values(['date', 'store_id', 'product_id']).reset_index(drop=True)
    
    print(f"✅ Dataset generated: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Regions: {df['region'].nunique()} | Stores: {df['store_id'].nunique()} | Products: {df['product_id'].nunique()}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  GCC Retail Sales Data Generator")
    print("=" * 60)
    
    df = generate_sales_data(
        num_stores=50,
        num_products=500,
        start_date='2020-01-01',
        end_date='2024-12-31',
        products_per_store=50,
        seed=42
    )
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sales_data.csv')
    print(f"\n💾 Saving to {save_path}...")
    df.to_csv(save_path, index=False)
    print(f"   File size: {os.path.getsize(save_path) / 1e6:.1f} MB")
    
    # Quick validation
    print("\n📊 Quick Validation:")
    print(f"   Zero stock → zero sales check: {(df[df['stock_available']==0]['units_sold']==0).all()}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Sample rows:")
    print(df.head(3).to_string(index=False))
