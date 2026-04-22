"""
Analyze sector distribution in the dataset and identify tech companies
"""
import pandas as pd
import numpy as np
from config import Config

def analyze_sectors():
    print("=== Dataset Sector Analysis ===\n")
    
    cfg = Config()
    df = pd.read_csv(cfg.DATA_PATH)
    
    print(f"Total companies in dataset: {len(df)}")
    print(f"Total unique companies: {df['Name'].nunique()}")
    
    # Check if Sector column exists
    if 'Sector' in df.columns:
        print("\n--- Sector Distribution ---")
        sector_counts = df['Sector'].value_counts()
        print(sector_counts)
        
        print("\n--- Sector Percentages ---")
        sector_pct = (sector_counts / len(df) * 100).round(2)
        for sector, pct in sector_pct.items():
            print(f"{sector:30s}: {pct:6.2f}%")
        
        # Identify tech-related sectors
        tech_keywords = ['Technology', 'Software', 'Computer', 'Electronic', 'Semiconductor', 
                        'Internet', 'Telecom', 'IT', 'Tech']
        
        tech_mask = df['Sector'].str.contains('|'.join(tech_keywords), case=False, na=False)
        tech_companies = df[tech_mask]
        
        print(f"\n--- Technology Sector Companies ---")
        print(f"Tech companies found: {len(tech_companies)}")
        print(f"Percentage: {len(tech_companies)/len(df)*100:.2f}%")
        
        if len(tech_companies) > 0:
            print(f"\nSample tech companies:")
            print(tech_companies[['Name', 'Sector', 'Rating']].drop_duplicates('Name').head(10))
            
            # Rating distribution for tech companies
            print(f"\n--- Tech Sector Rating Distribution ---")
            tech_ratings = tech_companies['Rating'].value_counts().sort_index()
            print(tech_ratings)
        
        # Save tech companies to file
        tech_companies.to_csv('tech_companies_in_dataset.csv', index=False)
        print(f"\nTech companies saved to: tech_companies_in_dataset.csv")
        
    else:
        print("\nWarning: 'Sector' column not found in dataset")
        print("Available columns:", df.columns.tolist())
    
    # Analyze rating distribution
    print("\n--- Overall Rating Distribution ---")
    rating_counts = df['Rating'].value_counts().sort_index()
    print(rating_counts)
    
    return df

if __name__ == "__main__":
    df = analyze_sectors()
