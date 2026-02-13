"""
PDF Report Generator for Crop Recommendation System
Creates professional PDF reports with recommendation details.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import io

class CropReportGenerator:
    """Generate professional PDF reports for crop recommendations."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#10b981'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#3b82f6'),
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#1f2937'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#374151'),
            spaceAfter=8,
            alignment=TA_JUSTIFY
        ))
    
    def generate_report(self, prediction_data, user_inputs):
        """
        Generate a PDF report for the crop recommendation.
        
        Args:
            prediction_data: Dictionary containing prediction results
            user_inputs: Dictionary containing user's input values
            
        Returns:
            BytesIO object containing the PDF
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        story = []
        
        # Add header
        story.extend(self._create_header())
        
        # Add recommendation summary
        story.extend(self._create_recommendation_summary(prediction_data))
        
        # Add ensemble details
        story.extend(self._create_ensemble_details(prediction_data))
        
        # Add input details
        story.extend(self._create_input_table(user_inputs))
        
        # Add reasoning
        if 'reasoning' in prediction_data:
            story.extend(self._create_reasoning_section(prediction_data['reasoning']))
        
        # Add influencing factors
        if 'top_influencing_factors' in prediction_data:
            story.extend(self._create_factors_section(prediction_data['top_influencing_factors']))
        
        # Add crop requirements
        if 'crop_info' in prediction_data:
            story.extend(self._create_crop_info_section(prediction_data['crop_info']))
        
        # Add footer
        story.extend(self._create_footer())
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _create_header(self):
        """Create the report header."""
        elements = []
        
        # Title
        title = Paragraph("🌾 Crop Recommendation Report", self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Date
        date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        date_para = Paragraph(f"<i>Generated on: {date_str}</i>", 
                             self.styles['Normal'])
        elements.append(date_para)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_recommendation_summary(self, data):
        """Create the main recommendation summary box."""
        elements = []
        
        crop_name = data.get('crop', 'Unknown').title()
        confidence = data.get('confidence', 0)
        agreement = data.get('agreement', 0)
        
        # Create a colored box for the recommendation
        summary_data = [
            ['Recommended Crop:', crop_name],
            ['Confidence Level:', f'{confidence}%'],
            ['Model Agreement:', f'{agreement}% ({data.get("vote_count", "N/A")})'],
            ['Decision Type:', '🏆 Unanimous' if data.get('unanimous') else '✨ Majority']
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ecfdf5')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#065f46')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#10b981')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#10b981')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#ecfdf5'), colors.HexColor('#d1fae5')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_ensemble_details(self, data):
        """Create ensemble model predictions table."""
        elements = []
        
        if 'individual_predictions' not in data:
            return elements
        
        header = Paragraph("Individual Model Predictions", self.styles['SectionHeader'])
        elements.append(header)
        
        # Create table data
        table_data = [['Model Name', 'Predicted Crop', 'Confidence']]
        
        for model_name, prediction in data['individual_predictions'].items():
            formatted_name = model_name.replace('_', ' ').title()
            crop = prediction['crop'].title()
            conf = f"{prediction['confidence']}%"
            table_data.append([formatted_name, crop, conf])
        
        model_table = Table(table_data, colWidths=[2*inch, 2*inch, 2*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')])
        ]))
        
        elements.append(model_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_input_table(self, user_inputs):
        """Create table showing user inputs."""
        elements = []
        
        header = Paragraph("Your Input Parameters", self.styles['SectionHeader'])
        elements.append(header)
        
        param_names = {
            'nitrogen': ('Nitrogen (N)', 'ratio'),
            'phosphorus': ('Phosphorus (P)', 'ratio'),
            'potassium': ('Potassium (K)', 'ratio'),
            'temperature': ('Temperature', '°C'),
            'humidity': ('Humidity', '%'),
            'ph': ('pH Level', 'pH'),
            'rainfall': ('Rainfall', 'mm')
        }
        
        table_data = [['Parameter', 'Value', 'Unit']]
        
        for key, value in user_inputs.items():
            if key in param_names:
                name, unit = param_names[key]
                table_data.append([name, str(value), unit])
        
        input_table = Table(table_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        input_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ede9fe')])
        ]))
        
        elements.append(input_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_reasoning_section(self, reasoning):
        """Create the reasoning/explanation section."""
        elements = []
        
        header = Paragraph("🎯 Why This Crop?", self.styles['SectionHeader'])
        elements.append(header)
        
        for reason in reasoning:
            bullet = Paragraph(f"• {reason}", self.styles['CustomBody'])
            elements.append(bullet)
        
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _create_factors_section(self, factors):
        """Create influencing factors section."""
        elements = []
        
        header = Paragraph("📊 Top Influencing Factors", self.styles['SectionHeader'])
        elements.append(header)
        
        table_data = [['Factor', 'Importance']]
        
        for factor in factors:
            table_data.append([factor['factor'], f"{factor['importance']}%"])
        
        factors_table = Table(table_data, colWidths=[3*inch, 3*inch])
        factors_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f59e0b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef3c7')])
        ]))
        
        elements.append(factors_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_crop_info_section(self, crop_info):
        """Create crop requirements and information section."""
        elements = []
        
        header = Paragraph("🌱 Crop Requirements & Information", self.styles['SectionHeader'])
        elements.append(header)
        
        # Description
        if 'description' in crop_info:
            desc = Paragraph(crop_info['description'], self.styles['CustomBody'])
            elements.append(desc)
            elements.append(Spacer(1, 10))
        
        # Growing conditions
        if 'growing_conditions' in crop_info:
            conditions_header = Paragraph("<b>Growing Conditions:</b>", self.styles['CustomBody'])
            elements.append(conditions_header)
            
            for condition in crop_info['growing_conditions']:
                bullet = Paragraph(f"  ✓ {condition}", self.styles['CustomBody'])
                elements.append(bullet)
            
            elements.append(Spacer(1, 10))
        
        # Metadata table
        metadata_data = [
            ['Season:', crop_info.get('season', 'N/A')],
            ['Water Requirement:', crop_info.get('water_requirement', 'N/A')],
            ['Climate:', crop_info.get('climate', 'N/A')]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef3c7')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#f59e0b')),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(metadata_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_footer(self):
        """Create report footer."""
        elements = []
        
        elements.append(Spacer(1, 30))
        
        footer_text = """
        <i>This report was generated by the Crop Recommendation System using ensemble machine learning 
        models (Random Forest, Gradient Boosting, and Decision Tree). The recommendations are based 
        on soil nutrient levels and weather conditions. For best results, consult with local 
        agricultural experts and consider regional farming practices.</i>
        """
        
        footer = Paragraph(footer_text, self.styles['Normal'])
        elements.append(footer)
        
        return elements
