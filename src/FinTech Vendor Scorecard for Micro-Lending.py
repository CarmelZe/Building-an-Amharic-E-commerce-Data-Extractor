#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EthioMart Vendor Scorecard System for Micro-Lending
Task 6: FinTech Vendor Scorecard for Micro-Lending
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os

class VendorScorecard:
    def __init__(self, vendor_data_path: str = None, ner_model_path: str = None):
        """
        Initialize the Vendor Scorecard system.
        
        Args:
            vendor_data_path: Path to JSON/CSV containing vendor post data
            ner_model_path: Path to fine-tuned NER model (optional)
        """
        # Load vendor data
        if vendor_data_path and os.path.exists(vendor_data_path):
            self.load_vendor_data(vendor_data_path)
        else:
            self.vendor_data = self.sample_data()  # Fallback to sample data
            
        # Initialize NER pipeline if model path provided
        self.ner_pipeline = None
        if ner_model_path:
            from transformers import pipeline
            self.ner_pipeline = pipeline("ner", 
                                       model=ner_model_path,
                                       tokenizer=ner_model_path,
                                       aggregation_strategy="simple")
    
    def load_vendor_data(self, file_path: str):
        """Load vendor data from JSON or CSV file"""
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.vendor_data = json.load(f)
        elif file_path.endswith('.csv'):
            self.vendor_data = pd.read_csv(file_path).to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")
    
    def sample_data(self) -> List[Dict]:
        """Generate sample vendor data for demonstration"""
        return [
            {
                "vendor_id": "vendor_001",
                "vendor_name": "Shager Online Store",
                "posts": [
                    {
                        "post_id": "post_001",
                        "text": "አዲስ ሳምሱንግ ስልክ በ15,000 ብር በአዲስ አበባ ይገኛል! ለተጨማሪ መረጃ ይደውሉ። 0912345678",
                        "views": 1250,
                        "timestamp": "2025-06-20T10:30:00",
                        "media_type": "text"
                    },
                    {
                        "post_id": "post_002",
                        "text": "የቤት እቃዎች ስብስብ በ50,000 ብር. ነፃ ማድረሻ! #AddisAbaba #HomeAppliances",
                        "views": 980,
                        "timestamp": "2025-06-18T14:15:00",
                        "media_type": "image"
                    }
                ]
            },
            {
                "vendor_id": "vendor_002",
                "vendor_name": "Bole Fashion Hub",
                "posts": [
                    {
                        "post_id": "post_003",
                        "text": "አዲስ የሴቶች ልብስ በ1,200 ብር ብቻ! በቦሌ ሚካኤል ሻንጣ ይገኛል።",
                        "views": 3200,
                        "timestamp": "2025-06-19T09:45:00",
                        "media_type": "text"
                    },
                    {
                        "post_id": "post_004",
                        "text": "የወንዶች ሸሚዝ በ800 ብር. ለ3 ቀናት ብቻ!",
                        "views": 1500,
                        "timestamp": "2025-06-17T11:20:00",
                        "media_type": "image"
                    }
                ]
            }
        ]
    
    def extract_entities(self, text: str) -> Dict:
        """Extract business entities using NER model"""
        if not self.ner_pipeline:
            return {"products": [], "prices": [], "locations": []}
            
        entities = self.ner_pipeline(text)
        result = {"products": [], "prices": [], "locations": []}
        
        for entity in entities:
            if entity['entity_group'] == 'PRODUCT':
                result['products'].append(entity['word'])
            elif entity['entity_group'] == 'PRICE':
                result['prices'].append(entity['word'])
            elif entity['entity_group'] == 'LOC':
                result['locations'].append(entity['word'])
        
        return result
    
    def calculate_metrics(self, vendor: Dict) -> Dict:
        """Calculate all metrics for a single vendor"""
        posts = vendor['posts']
        
        # Convert timestamps to datetime objects
        for post in posts:
            post['datetime'] = datetime.fromisoformat(post['timestamp'])
        
        # Sort posts by date
        posts.sort(key=lambda x: x['datetime'])
        
        # Calculate time range
        if len(posts) > 1:
            time_span = (posts[-1]['datetime'] - posts[0]['datetime']).days
            time_span_weeks = max(1, time_span / 7)  # Avoid division by zero
        else:
            time_span_weeks = 1  # Default to 1 week for single post
            
        # Initialize metrics
        metrics = {
            'vendor_id': vendor['vendor_id'],
            'vendor_name': vendor['vendor_name'],
            'total_posts': len(posts),
            'posting_frequency': len(posts) / time_span_weeks,
            'average_views': np.mean([post['views'] for post in posts]),
            'max_views': max(post['views'] for post in posts),
            'products': [],
            'prices': [],
            'top_post': None
        }
        
        # Process each post to extract entities and find top post
        top_post = None
        for post in posts:
            entities = self.extract_entities(post['text'])
            metrics['products'].extend(entities['products'])
            metrics['prices'].extend(entities['prices'])
            
            # Track top performing post
            if not top_post or post['views'] > top_post['views']:
                top_post = {
                    'post_id': post['post_id'],
                    'views': post['views'],
                    'text': post['text'],
                    'products': entities['products'],
                    'prices': entities['prices'],
                    'date': post['datetime'].strftime('%Y-%m-%d')
                }
        
        metrics['top_post'] = top_post
        
        # Calculate average price (convert price strings to numbers)
        numeric_prices = []
        for price in metrics['prices']:
            try:
                # Clean price string (remove currency symbols, commas)
                clean_price = ''.join(c for c in price if c.isdigit())
                if clean_price:
                    numeric_prices.append(float(clean_price))
            except:
                continue
        
        metrics['average_price'] = np.mean(numeric_prices) if numeric_prices else 0
        metrics['price_range'] = (min(numeric_prices), max(numeric_prices)) if numeric_prices else (0, 0)
        
        # Calculate lending score (custom weighted formula)
        metrics['lending_score'] = self.calculate_lending_score(metrics)
        
        return metrics
    
    def calculate_lending_score(self, metrics: Dict) -> float:
        """
        Calculate a composite lending score (0-100) based on business metrics.
        Customize these weights based on EthioMart's priorities.
        """
        # Normalize metrics (using min-max scaling for demonstration)
        max_views = max(v['average_views'] for v in self.vendor_data) if hasattr(self, 'vendor_data') else 5000
        max_frequency = max(v['posting_frequency'] for v in self.vendor_data) if hasattr(self, 'vendor_data') else 20
        
        normalized_views = min(metrics['average_views'] / max_views, 1.0)
        normalized_frequency = min(metrics['posting_frequency'] / max_frequency, 1.0)
        
        # Price stability factor (vendors with mid-range prices get higher scores)
        avg_price = metrics['average_price']
        price_stability = 0.5  # Default
        if avg_price > 0:
            if avg_price < 1000:
                price_stability = 0.7  # High volume, low margin
            elif 1000 <= avg_price <= 10000:
                price_stability = 0.9  # Ideal range
            else:
                price_stability = 0.6  # High margin, low volume
        
        # Weighted score calculation
        score = (
            (normalized_views * 0.4) + 
            (normalized_frequency * 0.3) + 
            (price_stability * 0.3)
        
        return round(score * 100, 2)
    
    def generate_scorecards(self) -> List[Dict]:
        """Generate scorecards for all vendors"""
        return [self.calculate_metrics(vendor) for vendor in self.vendor_data]
    
    def generate_report(self, scorecards: List[Dict], output_file: str = "vendor_scorecard_report.html"):
        """Generate an HTML report with visualizations"""
        # Create DataFrame for tabular display
        df = pd.DataFrame([{
            'Vendor ID': sc['vendor_id'],
            'Vendor Name': sc['vendor_name'],
            'Avg. Views/Post': round(sc['average_views'], 1),
            'Posts/Week': round(sc['posting_frequency'], 1),
            'Avg. Price (ETB)': round(sc['average_price'], 2),
            'Lending Score': sc['lending_score']
        } for sc in scorecards])
        
        # Sort by lending score
        df = df.sort_values('Lending Score', ascending=False)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>EthioMart Vendor Scorecard Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .high-score {{ background-color: #d4edda; }}
                .medium-score {{ background-color: #fff3cd; }}
                .low-score {{ background-color: #f8d7da; }}
                .chart {{ width: 100%; max-width: 600px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>EthioMart Vendor Scorecard Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Vendor Performance Summary</h2>
            {df.to_html(classes='scorecard-table', index=False, escape=False)}
            
            <h2>Top Performing Vendors</h2>
            <div class="chart">
                {self.generate_score_chart(scorecards)}
            </div>
            
            <h2>Detailed Vendor Analysis</h2>
        """
        
        # Add details for each vendor
        for sc in sorted(scorecards, key=lambda x: x['lending_score'], reverse=True):
            html_content += f"""
            <div class="vendor-detail">
                <h3>{sc['vendor_name']} (Score: {sc['lending_score']})</h3>
                <p><strong>Posting Frequency:</strong> {round(sc['posting_frequency'], 1)} posts/week</p>
                <p><strong>Average Engagement:</strong> {round(sc['average_views'], 0)} views per post</p>
                <p><strong>Price Range:</strong> ETB {round(sc['price_range'][0], 2)} - ETB {round(sc['price_range'][1], 2)}</p>
                
                <h4>Top Performing Post</h4>
                <p><strong>Views:</strong> {sc['top_post']['views']}</p>
                <p><strong>Date:</strong> {sc['top_post']['date']}</p>
                <p><strong>Content:</strong> {sc['top_post']['text']}</p>
                <p><strong>Products Mentioned:</strong> {', '.join(sc['top_post']['products']) or 'None detected'}</p>
                <p><strong>Prices Mentioned:</strong> {', '.join(sc['top_post']['prices']) or 'None detected'}</p>
            </div>
            <hr>
            """
        
        html_content += """
            </body>
            </html>
        """
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Vendor scorecard report generated: {output_file}")
    
    def generate_score_chart(self, scorecards: List[Dict]) -> str:
        """Generate a bar chart of vendor scores (returns HTML)"""
        import base64
        from io import BytesIO
        
        # Prepare data
        vendors = [sc['vendor_name'] for sc in scorecards]
        scores = [sc['lending_score'] for sc in scorecards]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        bars = plt.barh(vendors, scores, color='#3498db')
        plt.xlabel('Lending Score (0-100)')
        plt.title('Vendor Lending Scores')
        plt.xlim(0, 100)
        
        # Add score labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width - 5, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}', 
                    ha='center', va='center', color='white')
        
        # Save to temporary buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Convert to base64 for HTML embedding
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f'<img src="data:image/png;base64,{data}" alt="Vendor Scores Chart"/>'

def main():
    # Initialize the scorecard system
    print("Initializing EthioMart Vendor Scorecard System...")
    
    # Example usage (replace with actual data path)
    scorecard_system = VendorScorecard(
        vendor_data_path="vendor_data.json",  # Replace with your data file
        ner_model_path="../model/amharic_ner_task3.ipynb"  # Optional
    )
    
    # Generate scorecards
    print("Calculating vendor metrics...")
    scorecards = scorecard_system.generate_scorecards()
    
    # Generate report
    print("Generating vendor scorecard report...")
    scorecard_system.generate_report(scorecards)
    
    print("Vendor scorecard analysis completed successfully!")

if __name__ == "__main__":
    main()