"""
PDF Report Generator for RecoMart Recommendation System Pipeline.
Generates PDF deliverables for Problem Formulation, Data Quality, and Model Performance.
"""

import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib import colors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_pdf_report(filename, title, subtitle, sections, output_dir="reports"):
    """
    Creates a styled PDF report.
    :param filename: Output PDF file name (e.g. 'problem_formulation.pdf')
    :param title: Report main title string
    :param subtitle: Subtitle string
    :param sections: List of dicts with 'heading', 'content' (paragraph string or table or list of strings), and optional 'table_data' or 'image_path'
    :param output_dir: Directory where PDF will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E40AF'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=8
    )

    elements = []
    
    # Title & Header
    elements.append(Paragraph(title, title_style))
    elements.append(Paragraph(subtitle, subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3B82F6'), spaceAfter=12))

    for section in sections:
        if 'heading' in section:
            elements.append(Paragraph(section['heading'], h1_style))
        
        if 'content' in section:
            if isinstance(section['content'], str):
                elements.append(Paragraph(section['content'], body_style))
            elif isinstance(section['content'], list):
                for p in section['content']:
                    elements.append(Paragraph(p, body_style))
        
        if 'table_data' in section:
            t = Table(section['table_data'], hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2563EB')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 9),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,1), (-1,-1), 8),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 8))

        if 'image_path' in section and os.path.exists(section['image_path']):
            elements.append(Spacer(1, 4))
            elements.append(Image(section['image_path'], width=450, height=220))
            elements.append(Spacer(1, 8))
            
        elements.append(Spacer(1, 6))

    doc.build(elements)
    logging.info(f"Generated PDF report successfully at '{pdf_path}'")
    return pdf_path


def generate_problem_formulation_pdf(output_dir="reports"):
    title = "RecoMart Recommendation System: Problem Formulation & Architecture"
    subtitle = "Prepared by RecoMart Data Platform Team | Data Engineering & ML Architecture"
    
    sections = [
        {
            "heading": "1. Business Objective & Context",
            "content": "RecoMart is a rapidly growing e-commerce platform aiming to maximize user engagement, click-through rates (CTR), and average order value (AOV) via hyper-personalized product recommendations. By deploying a robust end-to-end data and ML pipeline, RecoMart can systematically ingest multi-source user interactions, curate validated features, and continuously update recommendation models."
        },
        {
            "heading": "2. Data Sources & Attributes",
            "table_data": [
                ["Data Source", "Format / Ingestion Mode", "Key Attributes Description"],
                ["Clickstream & Transactions", "CSV (Batch Ingestion)", "user_id, item_id, rating (1.0-5.0), interaction_type (view, cart, purchase), timestamp, device"],
                ["Product Metadata", "REST API (Batch/Near-RT)", "item_id, product_name, category, brand, price, sentiment_score, popularity_score"],
                ["User Demographics", "CSV / DB", "user_id, age, gender, signup_date, membership_tier"]
            ]
        },
        {
            "heading": "3. Target Pipeline Outputs",
            "content": [
                "<b>• Raw Data Lake:</b> Partitioned storage by data source, year, month, and day.",
                "<b>• Validated Datasets:</b> Automated quality reports flagging nulls, schema drifts, and out-of-bound ratings.",
                "<b>• Feature Store:</b> Standardized user and item offline/online feature views for zero-leakage training.",
                "<b>• Trained Recommendation Models:</b> Collaborative filtering (SVD) and content-based recommendation artifacts tracked via MLflow.",
                "<b>• Inference API:</b> High-throughput scoring interface for real-time recommendation retrieval."
            ]
        },
        {
            "heading": "4. Evaluation Metrics",
            "table_data": [
                ["Metric", "Category", "Target Benchmark / Description"],
                ["Precision@K", "Ranking Quality", "Fraction of recommended items in Top-K that are relevant to the user."],
                ["Recall@K", "Coverage", "Fraction of total relevant items retrieved within Top-K."],
                ["NDCG@K", "Ranking Order", "Normalized Discounted Cumulative Gain accounting for position decay."],
                ["RMSE / MAE", "Rating Error", "Root Mean Squared Error & Mean Absolute Error on predicted rating scores."]
            ]
        }
    ]
    return create_pdf_report("problem_formulation.pdf", title, subtitle, sections, output_dir)
