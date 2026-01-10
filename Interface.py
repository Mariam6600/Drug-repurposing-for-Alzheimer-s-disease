# Import required libraries
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, RGATConv
from torch_geometric.data import Data
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import io
import google.generativeai as genai
import json
from dotenv import load_dotenv

# Load environment variables from multiple possible locations
load_dotenv()
load_dotenv(r"C:\Users\LOQ\Desktop\Graduation\.env")  # Explicit path

def translate_arabic_to_english(arabic_text, drug_name, gene_name, relation, confidence):
    """Translate Arabic scientific explanation to English while preserving accuracy and meaning"""
    
    # If text is already mostly English, just clean it
    arabic_chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
    arabic_count = sum(1 for char in arabic_text if char in arabic_chars)
    
    # If less than 10% Arabic characters, treat as English text that needs cleaning
    if arabic_count < len(arabic_text) * 0.1:
        # Clean English text and ensure it's properly formatted
        english_text = arabic_text
        # Remove any remaining Arabic punctuation
        english_text = english_text.replace('،', ',').replace('؛', ';').replace('؟', '?')
        english_text = english_text.replace('٪', '%')
        return english_text
    
    # Common Arabic-English scientific translations
    translations = {
        # Basic terms
        'بناءً على البيانات المتاحة': 'Based on available data',
        'النموذج الحاسوبي يقترح': 'the computational model suggests',
        'النموذج الحاسوبي يشير إلى': 'the computational model indicates',
        'التحليل يشير إلى إمكانية': 'the analysis indicates the possibility',
        'هذا النظام مصمم لاستكشاف التفاعلات الدوائية-الجينية': 'This system is designed to explore drug-gene interactions',
        'في سياق الأبحاث العصبية': 'in neuroscience research context',
        'جميع التنبؤات حاسوبية': 'all predictions are computational',
        'تتطلب التحقق التجريبي': 'require experimental validation',
        
        # Drug-Gene Analysis
        'تحليل التفاعل الدوائي-الجيني': 'Drug-Gene Interaction Analysis',
        'الدواء': 'Drug',
        'الجين': 'Gene',
        'نوع التفاعل المتوقع': 'Predicted Interaction Type',
        'مستوى الثقة': 'Confidence Level',
        
        # Scientific explanation terms
        'التفسير العلمي': 'Scientific Explanation',
        'احتمالية حدوث تفاعل': 'potential interaction occurrence',
        'قد يكون ذا صلة': 'may be relevant',
        'قد ترتبط بالتعبير الجيني': 'may be associated with gene expression',
        'قد تتفاعل مع البروتينات': 'may interact with proteins',
        'المُشفرة من هذا الجين': 'encoded by this gene',
        'ضمن نطاق هذا التحليل': 'within the scope of this analysis',
        'يشير مستوى الثقة إلى قوة الدليل الحاسوبي': 'the confidence level indicates the strength of computational evidence',
        'للتفاعل المتوقع': 'for the predicted interaction',
        
        # Advanced terms
        'الأدوية من هذه الفئة': 'drugs of this class',
        'قد يرتبط بـ': 'may be associated with',
        'قد يكون مرتبطًا بتقليل': 'may be associated with reduced',
        'قد يكون مرتبطًا بزيادة': 'may be associated with increased',
        'النموذج الحاسوبي يتوقع ارتباطًا محتملاً': 'the computational model predicts a potential association',
        'التمثيلات الرقمية': 'digital representations',
        'التشابه الحاسوبي': 'computational similarity',
        'ارتباط وظيفي محتمل': 'potential functional association',
        'يستند هذا التنبؤ إلى': 'this prediction is based on',
        'يعتمد هذا التنبؤ على': 'this prediction relies on',
        'مما يشير إلى': 'which indicates',
        'مما يقترح وجود': 'suggesting the presence of',
        'أساس بيولوجي محتمل': 'potential biological basis',
        'ملفات العقار والجين': 'drug and gene profiles',
        'بين التمثيلات الرقمية للعقار والجين': 'between the digital representations of the drug and gene',
        
        # Biological processes
        'موت الخلايا المبرمج': 'programmed cell death (apoptosis)',
        'الاستماتة': 'apoptosis',
        'التنظيم السلبي للاستماتة': 'negative regulation of apoptosis',
        'التنظيم الإيجابي لتكاثر الخلايا': 'positive regulation of cell proliferation',
        'بقاء الخلية': 'cell survival',
        'تكاثرها': 'proliferation',
        'نمو الخلايا': 'cell growth',
        'استقلاب الستيرول': 'sterol metabolism',
        'استقلاب المواد الغريبة': 'xenobiotic metabolism',
        'الإجهاد التأكسدي': 'oxidative stress',
        'الالتهام الذاتي': 'autophagy',
        'تطور الأوعية الدموية': 'angiogenesis',
        
        # Molecular functions
        'الوظائف الجزيئية': 'molecular functions',
        'نشاط الأكسدة والاختزال': 'oxidoreductase activity',
        'نشاط أحادي الأكسجة': 'monooxygenase activity',
        'نشاط اختزال الأكسدة': 'oxidoreductase activity',
        'نشاط إنزيم هيم أوكسيجيناز': 'heme oxygenase activity',
        'نشاطًا مثبطًا': 'inhibitory activity',
        'وظائف ارتباط نطاق BH3': 'BH3 domain binding functions',
        'تفاعلات البروتين-البروتين': 'protein-protein interactions',
        'المسارات التنظيمية': 'regulatory pathways',
        
        # Drug classes
        'مثبطات اختزال الإنزيم': 'enzyme reductase inhibitors',
        'مثبطات': 'inhibitors',
        'ضمن': 'within',
        'يُصنَّف': 'is classified',
        'يُصنف': 'is classified',
        'يُعرف': 'is known',
        
        # Warnings and conclusions
        'تتطلب هذه النتائج الحاسوبية الأولية': 'These preliminary computational results require',
        'تحققًا تجريبيًا دقيقًا': 'rigorous experimental validation',
        'للتأكد من إمكانياتها العلاجية': 'to confirm their therapeutic potential',
        'قبل أي تطبيق سريري': 'before any clinical application',
        'تأكيد أي علاقة بيولوجية حقيقية': 'confirm any real biological relationship',
        'تحديد الآليات الكامنة': 'determine the underlying mechanisms',
        'والبحث المخبري': 'and laboratory research',
        
        # Academic improvements based on ChatGPT feedback
        'ارتباط وظيفي محتمل': 'potential functional association',
        'قد تكون مرتبطة وظيفيًا': 'may be functionally associated',
        'يستند هذا التنبؤ إلى أنماط التشابه الحاسوبي': 'this prediction is based on computational similarity patterns',
        'في فضاء التمثيل': 'in representation space',
        'خصائص كامنة مشتركة': 'shared latent characteristics',
        'النتائج الحاسوبية الاستكشافية': 'exploratory computational predictions',
        'تحققًا تجريبيًا دقيقًا': 'rigorous experimental validation',
        
        # Improved causality language
        'احتمالية حدوث تفاعل': 'potential association occurrence',
        'وجود ارتباط وظيفي محتمل': 'potential functional association',
        'قد يكون مرتبطًا وظيفيًا بـ': 'may be functionally associated with',
        'انخفاض محتمل في تعبير': 'potential decrease in expression of',
        'زيادة محتملة في تعبير': 'potential increase in expression of',
        'ضمن سياق استقلاب الدواء': 'within the context of drug metabolism',
        'استنادًا إلى وظائف الإنزيم': 'based on enzyme functions',
        'واستقلاب المركبات المختلفة': 'and metabolism of various compounds',
        'واستقلاب المركبات المتعددة': 'and metabolism of multiple compounds',
        'في الأكسدة والاختزال': 'in oxidation and reduction',
        'سمات الفئات الدوائية والمسارات البيولوجية': 'characteristics of drug classes and biological pathways',
        'التي تشترك فيها هذه الكيانات': 'shared by these entities',
        'أهميتها البيولوجية والفارماكولوجية': 'their biological and pharmacological significance',
        'تأكيد هذه التفاعلات المحتملة': 'confirm these potential interactions',
        
        # BCL2 specific terms
        'بانخفاض تعبير الجين': 'with decreased gene expression',
        'بتقليل تعبير الجين': 'with reduced gene expression',
        'الانخفاض المقترح في تعبير': 'the proposed decrease in expression of',
        'الانخفاض المحتمل في تعبير': 'the potential decrease in expression of',
        'تأثيرات محتملة على بقاء الخلية أو تكاثرها': 'potential effects on cell survival or proliferation',
        'تأثيرات على بقاء الخلية أو نموها': 'effects on cell survival or growth',
        'مما قد يعزز الاستماتة': 'which may promote apoptosis',
        'أو يقلل من نمو الخلايا': 'or reduce cell growth',
        'نشاطًا مثبطًا': 'inhibitory activity',
        'وظائف ارتباط نطاق BH3': 'BH3 domain binding functions',
        'ارتباطه بمجال BH3': 'its binding to BH3 domain',
        'ضمن المسارات التنظيمية للاستماتة': 'within apoptotic regulatory pathways',
        'ضمن مسارات الاستماتة التنظيمية': 'within regulatory apoptotic pathways',
        
        # Specific relation types
        'CHEMICALBINDSGENE': 'chemical-gene binding',
        'CHEMICALINCREASESEXPRESSION': 'chemical increases gene expression',
        'CHEMICALDECREASESEXPRESSION': 'chemical decreases gene expression',
        'NO_LINK': 'no significant link'
    }
    
    # Start with the original text
    english_text = arabic_text
    
    # Apply translations
    for arabic, english in translations.items():
        english_text = english_text.replace(arabic, english)
    
    # Handle specific drug and gene names (keep as is)
    # Handle confidence percentage format
    english_text = english_text.replace('٪', '%')
    
    # Clean up any remaining Arabic punctuation
    english_text = english_text.replace('،', ',')
    english_text = english_text.replace('؛', ';')
    english_text = english_text.replace('؟', '?')
    
    # Handle section headers
    english_text = english_text.replace('🔬 **تحليل التفاعل الدوائي-الجيني**', '🔬 **Drug-Gene Interaction Analysis**')
    english_text = english_text.replace('**التفسير العلمي:**', '**Scientific Explanation:**')
    english_text = english_text.replace('**الدواء:**', '**Drug:**')
    english_text = english_text.replace('**الجين:**', '**Gene:**')
    english_text = english_text.replace('**نوع التفاعل المتوقع:**', '**Predicted Interaction:**')
    english_text = english_text.replace('**مستوى الثقة:**', '**Confidence Level:**')
    
    # If still contains significant Arabic after translation, provide fallback
    remaining_arabic = sum(1 for char in english_text if char in arabic_chars)
    if remaining_arabic > len(english_text) * 0.05:  # If more than 5% still Arabic
        # Provide comprehensive English fallback
        english_text = f"""This analysis is based on a computational model for drug-gene relationship prediction in neuroscience research context.

**Drug-Gene Interaction Analysis**

**Drug:** {drug_name}
**Gene:** {gene_name}
**Predicted Interaction:** {relation}
**Confidence Level:** {confidence}

**Scientific Explanation:**
Based on available data, the computational model suggests a potential functional association between {drug_name} and {gene_name}. The analysis indicates this association may be relevant in neuroscience research context.

According to current data, drugs of this class may be functionally associated with gene expression changes or may interact with proteins encoded by this gene. This prediction is based on computational similarity patterns in representation space, indicating shared latent characteristics.

WARNING: These exploratory computational predictions require rigorous experimental validation and laboratory research to confirm their therapeutic potential before any clinical application."""
    
    return english_text

def generate_pdf_report(report_list, selected_lang):
    """Generate comprehensive PDF report from basket contents - ENGLISH ONLY - Each analysis on separate page"""
    buffer = io.BytesIO()
    
    # Create PDF
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Title and header - ALWAYS IN ENGLISH
    title = "Alzheimer's Drug Discovery Analysis Report"
    subtitle = f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
    summary_title = "Analysis Summary"
    
    # Header
    p.setFont("Helvetica-Bold", 18)
    p.drawString(50, height - 50, title)
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 75, subtitle)
    
    # Summary section
    y_position = height - 110
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y_position, summary_title)
    y_position -= 25
    
    p.setFont("Helvetica", 11)
    p.drawString(50, y_position, f"Total Predictions: {len(report_list)}")
    y_position -= 15
    
    # Count by relation type
    relation_counts = {}
    for report in report_list:
        rel = report['relation']
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    for relation, count in relation_counts.items():
        p.drawString(70, y_position, f"• {relation}: {count} predictions")
        y_position -= 15
    
    # Add space before first analysis
    y_position -= 30
    
    # Start detailed analyses - FIRST ONE ON SAME PAGE
    for i, report in enumerate(report_list):
        # Start new page for each analysis EXCEPT the first one
        if i > 0:
            p.showPage()
            y_position = height - 50
        else:
            # Check if we have enough space for first analysis on current page
            if y_position < 300:  # Need at least 300 points for analysis
                p.showPage()
                y_position = height - 50
        
        # Analysis header
        p.setFont("Helvetica-Bold", 16)
        analysis_header = f"Analysis {i+1}: {report['drug']} ↔ {report['gene']}"
        p.drawString(50, y_position, analysis_header)
        
        # Horizontal line under header
        p.line(50, y_position - 15, width - 50, y_position - 15)
        
        y_position -= 40
        
        # Basic details in a box
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y_position, "Prediction Details:")
        y_position -= 25
        
        p.setFont("Helvetica", 11)
        details = [
            f"Drug: {report['drug']}",
            f"Gene: {report['gene']}",
            f"Predicted Relation: {report['relation']}",
            f"Confidence Score: {report['prob']}"
        ]
        
        for detail in details:
            p.drawString(70, y_position, detail)
            y_position -= 18
        
        y_position -= 20  # Extra space before explanation
        
        # Explanation section
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y_position, "Scientific Explanation:")
        y_position -= 20
        
        # Clean explanation text and convert Arabic to English
        explanation_text = report['explanation']
        
        # Convert ALL Arabic explanations to English with accurate scientific translation
        if any(arabic_char in explanation_text for arabic_char in ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']):
            # Translate Arabic explanation to English while preserving scientific accuracy
            explanation_text = translate_arabic_to_english(explanation_text, report['drug'], report['gene'], report['relation'], report['prob'])
        else:
            # Even if it's already English, ensure it's clean for PDF
            explanation_text = explanation_text
        
        # Clean text for PDF - remove emojis and formatting but keep warning content
        explanation_text = explanation_text.replace('**', '').replace('*', '')
        explanation_text = explanation_text.replace('🤖', '').replace('🔬', '').replace('✅', '').replace('🏠', '')
        explanation_text = explanation_text.replace('AI Generated:', '').replace('Local Interpretation:', '')
        
        # Handle warning symbol - convert to text
        explanation_text = explanation_text.replace('⚠️', 'WARNING:')
        
        explanation_text = explanation_text.strip()
        
        # Process text with better formatting
        p.setFont("Helvetica", 10)
        lines = explanation_text.split('\n')
        
        for line in lines:
            if line.strip():
                # Word wrapping with better spacing
                words = line.strip().split(' ')
                current_line = ""
                
                for word in words:
                    test_line = current_line + word + " "
                    if len(test_line) < 75:  # Longer lines for better readability
                        current_line = test_line
                    else:
                        if current_line.strip():
                            # Check if we need new page
                            if y_position < 80:
                                p.showPage()
                                y_position = height - 50
                            p.drawString(70, y_position, current_line.strip())
                            y_position -= 14
                        current_line = word + " "
                
                # Write remaining text
                if current_line.strip():
                    if y_position < 80:
                        p.showPage()
                        y_position = height - 50
                    p.drawString(70, y_position, current_line.strip())
                    y_position -= 14
            else:
                y_position -= 8  # Empty line spacing
    
    # Footer on last page
    p.setFont("Helvetica", 8)
    p.drawString(50, 30, f"Generated by Alzheimer's Drug Discovery AI System")
    p.drawString(50, 20, "Note: All predictions are computational and require rigorous experimental validation")
    
    p.save()
    buffer.seek(0)
    return buffer


st.set_page_config(page_title="Alzheimer Discovery AI", layout="wide")

# Add CSS for RTL Arabic text and LTR AI header
st.markdown("""
<style>
.ai-header {
    direction: ltr !important;
    text-align: left !important;
    font-weight: bold;
    color: #1f77b4;
    margin-bottom: 10px;
    font-size: 16px;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 5px;
}
.arabic-content {
    direction: rtl;
    text-align: right;
}
.arabic-title {
    direction: rtl !important;
    text-align: right !important;
}
.english-title {
    direction: ltr !important;
    text-align: left !important;
}
.arabic-sidebar {
    direction: rtl !important;
    text-align: right !important;
}
.english-sidebar {
    direction: ltr !important;
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)

if 'report_list' not in st.session_state:
    st.session_state.report_list = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_explanation' not in st.session_state:
    st.session_state.current_explanation = None
if 'used_model' not in st.session_state:
    st.session_state.used_model = None

#=====================================================================
LANG = {
    "English": {
        "title": "Alzheimer's Drug Discovery",
        "select_drug": "Select a Drug:",
        "select_gene": "Select a Gene:",
        "select_relation": "Select Relation Type:",
        "predict_btn": "Predict Link",
        "explain_btn": "Explain with AI",
        "add_report": "Add to Report Basket",
        "export_pdf": "Download PDF Report",
        "stats_btn": "Model Stats",
        "reports_added": "Basket: ",
        "no_reports": "Basket is empty.",
        "result_pos": "Link Predicted!",
        "result_neg": "No Strong Link Found.",
        "clear_btn": "Clear Basket",
        "gene_label": "Gene",
        "status_ready": "System Ready"
    },

    "العربية": {
        "title": "اكتشاف أدوية الزهايمر",
        "select_drug": "اختر دواءً من القائمة:",
        "select_gene": "اختر الجين:",
        "select_relation": "اختر نوع العلاقة:",
        "predict_btn": "تنبؤ بوجود رابط",
        "explain_btn": "تفسير بالذكاء الاصطناعي",
        "add_report": "إضافة إلى سلة التقرير",
        "export_pdf": "تحميل ملف PDF المجمع",
        "stats_btn": "دقة النموذج",
        "reports_added": "السلة: ",
        "no_reports": "السلة فارغة.",
        "result_pos": "تم التنبؤ بوجود ارتباط!",
        "result_neg": "لا يوجد ارتباط قوي.",
        "clear_btn": "تفريغ السلة",
        "gene_label": "الجين",
        "status_ready": "النظام جاهز"
    }
}

st.sidebar.title("Settings")
selected_lang = st.sidebar.selectbox("Language", ["العربية", "English"])
texts = LANG[selected_lang]

# AI Configuration
USE_AI = True  # Set to True to enable AI explanations

# ALL available working models (complete list from API check)
AVAILABLE_GEMINI_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro", 
    "models/gemini-2.0-flash-exp",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-2.0-flash-lite",
    "models/gemini-exp-1206",
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest", 
    "models/gemini-pro-latest",
    "models/gemini-2.5-flash-lite",
    "models/gemini-3-pro-preview",
    "models/gemini-3-flash-preview"
]

# Load Gemini API Key from environment with fallback
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') if USE_AI else None

# If not found, try alternative names
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')

# If still not found, show warning (removed hardcoded key for security)
if not GEMINI_API_KEY:
    st.sidebar.warning("⚠️ Please add GEMINI_API_KEY to your .env file")

if GEMINI_API_KEY and USE_AI:
    genai.configure(api_key=GEMINI_API_KEY)
    st.sidebar.success("🤖 AI Provider: **Gemini** ✅")
elif USE_AI:
    st.sidebar.warning("🤖 AI Provider: **Gemini** (تحقق من المفتاح)")
else:
    st.sidebar.info("🤖 AI Provider: **Local Mode** (محلي)")

#===========================================================================================================


class Advanced_RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_rel):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_rel)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_rel)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def encode(self, x, edge_index, edge_type):
        h1 = self.conv1(x, edge_index, edge_type)
        h1 = self.ln1(torch.relu(h1))
        h1 = self.dropout(h1)
        
        h2 = self.conv2(h1, edge_index, edge_type)
        h2 = self.ln2(torch.relu(h2))
        
        return h1 + h2 

    def decode(self, h, edges):
        src = h[edges[:, 0]]
        dst = h[edges[:, 1]]
        z = torch.cat([src, dst], dim=1)
        return self.edge_mlp(z)

#  RGAT

class Advanced_RGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_rel, heads=4):
        super().__init__()
        self.conv1 = RGATConv(in_dim, hidden_dim // heads, num_rel, heads=heads)
        self.conv2 = RGATConv(hidden_dim, hidden_dim // heads, num_rel, heads=heads)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def encode(self, x, edge_index, edge_type):
        h1 = self.conv1(x, edge_index, edge_type)
        h1 = self.ln1(torch.relu(h1))
        h1 = self.dropout(h1)
        
        h2 = self.conv2(h1, edge_index, edge_type)
        h2 = self.ln2(torch.relu(h2))
        
        return h1 + h2

    def decode(self, h, edges):
        src = h[edges[:, 0]]
        dst = h[edges[:, 1]]
        z = torch.cat([src, dst], dim=1)
        return self.edge_mlp(z)

#============================================================================================================

@st.cache_resource
def load_resources():
    base_path = r"C:\Users\LOQ\Desktop\Graduation\\"  
    
    # Load tensors
    raw = torch.load(base_path + "alz_raw_tensors.pt", map_location='cpu')
    x = raw['x']
    edge_index = raw['edge_index']
    data = Data(x=x, edge_index=edge_index)
    data.edge_type = raw['edge_type'] if 'edge_type' in raw else torch.zeros(edge_index.size(1), dtype=torch.long)

    # Load drug list
    df_drugs = pd.read_csv(base_path + "alz_drugs_list.csv")
    col_name = 'drugName' if 'drugName' in df_drugs.columns else 'name'
    df_drugs = df_drugs.sort_values(by=col_name)
    drug_map = dict(zip(df_drugs[col_name], df_drugs['nodeID']))

    # Load gene list and create gene mapping
    df_genes = pd.read_csv(base_path + "alz_genes_list.csv")
    gene_id_to_symbol = dict(zip(df_genes['nodeID'].astype(str), df_genes['geneSymbol']))

    # Load node features
    df_features = pd.read_csv(base_path + "alz_node_features.csv")

     # Load metadata
    with open(base_path + "drug_metadata.json", "r", encoding="utf-8") as f:
        drug_metadata = json.load(f)

    with open(base_path + "gene_metadata.json", "r", encoding="utf-8") as f:
        gene_metadata = json.load(f)

    in_dim = data.x.size(1)    
    hidden_dim = 256
    num_classes = 4
    num_rel = 1
    heads = 4

    rgcn = Advanced_RGCN(in_dim, hidden_dim, num_classes, num_rel)
    rgat = Advanced_RGAT(in_dim, hidden_dim, num_classes, num_rel, heads=heads)


    rgcn.load_state_dict(torch.load(r"C:\Users\LOQ\Desktop\Graduation\rgcn_multi.pt", map_location='cpu'))
    rgat.load_state_dict(torch.load(r"C:\Users\LOQ\Desktop\Graduation\rgat_multi.pt", map_location='cpu'))

    return (
        data,
        df_features,
        drug_map,
        list(drug_map.keys()),
        rgcn.eval(),
        rgat.eval(),
        drug_metadata,     
        gene_metadata,
        gene_id_to_symbol  # Add gene mapping
    )

try:
    data, df_features, drug_map, drug_names, rgcn_model, rgat_model, drug_metadata, gene_metadata, gene_id_to_symbol = load_resources()
    # Models loaded successfully - no sidebar message
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {str(e)}")
    st.stop()

# =================================================================================================================
CLASS_NAMES = [
    "NO_LINK",
    "CHEMICALBINDSGENE",
    "CHEMICALINCREASESEXPRESSION",
    "CHEMICALDECREASESEXPRESSION"
]

def predict_interaction_with_embeddings(drug_id):
    node_to_idx = {str(n): i for i, n in enumerate(df_features["nodeID"].astype(str))}
    d_idx = node_to_idx[str(drug_id)]
    gene_indices = df_features[df_features['label_Gene'] == 1].index.tolist()
    edge_pairs = torch.tensor([[d_idx, g_idx] for g_idx in gene_indices], dtype=torch.long)
    with torch.no_grad():
        emb_rgcn = rgcn_model.encode(data.x, data.edge_index, data.edge_type)
        emb_rgat = rgat_model.encode(data.x, data.edge_index, data.edge_type)
        h_final = torch.cat([emb_rgcn, emb_rgat], dim=1)
        logits_rgcn = rgcn_model.decode(emb_rgcn, edge_pairs)  
        logits_rgat = rgat_model.decode(emb_rgat, edge_pairs) 
        logits = 0.7 * logits_rgcn + 0.3 * logits_rgat
        probs = torch.softmax(logits, dim=1)   

    results = []
    for i, g_idx in enumerate(gene_indices):
        class_probs = probs[i].tolist()               # [p0, p1, p2, p3]
        best_class = int(torch.argmax(probs[i]))      # index 0..3
        best_prob = class_probs[best_class]           # probability of best class

        results.append({
            "gene": df_features.iloc[g_idx]['nodeID'],
            "class_id": best_class,
            "class_name": CLASS_NAMES[best_class],
            "prob": best_prob,
            "all_probs": class_probs,
            "drug_emb": h_final[d_idx].tolist(),
            "gene_emb": h_final[g_idx].tolist()
        })
    results = sorted(results, key=lambda x: x['prob'], reverse=True)
    return results if len(results) > 0 else None

# ==================================================================================================================

def try_all_gemini_models(
        drug_name, drug_id, gene_name, gene_id,
        class_id, class_prob,
        drug_emb, gene_emb,
        drug_metadata, gene_metadata,
        lang):
    """Try all available Gemini models in order until one works"""
    
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    
    if not GEMINI_API_KEY:
        return None, "No API Key"
    
    relation_name = CLASS_NAMES[class_id]
    drug_info = drug_metadata.get(str(drug_id), {})
    gene_info = gene_metadata.get(str(gene_id), {})

    # Extract metadata
    drug_classes = ", ".join(drug_info.get("classes", [])) or "No known drug classes"
    
    bp = ", ".join(gene_info.get("biological_processes", [])) or "No known biological processes"
    mf = ", ".join(gene_info.get("molecular_functions", [])) or "No known molecular functions"
    cc = ", ".join(gene_info.get("cellular_components", [])) or "No known cellular components"

    # Set language
    llm_lang = "Arabic" if lang == "العربية" else "English"
    
    # Updated prompt with improved cautious language and terminology
    prompt = f"""
You are an expert biomedical AI explainer specializing in Alzheimer's disease research.

CRITICAL INSTRUCTIONS FOR ACADEMIC FRAMING:
1. Use cautious, academic language that distinguishes between computational predictions and biological validation
2. In Arabic, use "النموذج يقترح" (the model suggests) instead of "النموذج يشير إلى" (the model indicates)
3. Emphasize that this is a computational prediction requiring experimental validation
4. Use "بناءً على البيانات المتاحة" once at the beginning, avoid excessive repetition
5. Use "predicted to be associated with" instead of causative language like "increases/decreases"
6. Use "association" or "functional association" instead of "interaction" for sensitive cases
7. Do not include detailed vector numbers in explanations
8. Avoid repeating confidence scores multiple times
9. Use "suggests" instead of "indicates" for softer scientific language
10. Use "may be associated with" instead of "modulates" for weaker causal claims
11. Describe computational evidence as "similarity patterns in representation space" not "molecular similarity"
12. Emphasize predictions are "exploratory and hypothesis-generating" in nature

Your task is to explain the predicted relationship between a drug and a gene
using biological metadata, drug classes, gene functions, and computational similarity.

DRUG INFORMATION:
Drug Name: {drug_name}
Drug ID: {drug_id}
Drug Classes: {drug_classes}

GENE INFORMATION:
Gene Name: {gene_name}
Gene ID: {gene_id}
Biological Processes: {bp}
Molecular Functions: {mf}
Cellular Components: {cc}

MODEL PREDICTION:
Predicted Relation Type: {relation_name}
Confidence Score: {class_prob:.4f}

TASK:
Write a scientifically rigorous explanation (6-8 sentences) that:
1. Starts with methodological context (computational prediction)
2. Explains the predicted relationship using available drug classes and gene functions
3. Uses cautious language about biological plausibility
4. Discusses computational evidence as similarity patterns in representation space (without detailed numbers)
5. Adds appropriate protein-specific context when relevant (transporters, receptors, enzymes)
6. ⚠️ Concludes with clear statement that these computational predictions are exploratory and require rigorous experimental validation

LANGUAGE REQUIREMENTS:
- Write entirely in: {llm_lang}
- Use academic, cautious phrasing moderately (avoid excessive repetition)
- Base analysis ONLY on provided metadata
- If information is missing, state this explicitly
- For Arabic explanations: Use the Unified Medical Dictionary (القاموس الطبي الموحد) for translating scientific and medical terminology

IMPROVED TERMINOLOGY FOR RELATIONS:
- For CHEMICALINCREASESEXPRESSION: Use "The computational model predicts an association between [drug] and increased expression of [gene]"
- For CHEMICALDECREASESEXPRESSION: Use "The computational model predicts an association between [drug] and reduced expression of [gene]"
- For CHEMICALBINDSGENE: Use "The analysis suggests a potential functional association between [drug] and [gene]"

PROTEIN-SPECIFIC DISCLAIMERS:
- For transporters (e.g., TF): "functional association involving TF-mediated pathways"
- For nuclear receptors (e.g., PPARG): "potential functional association with PPARG signaling"
- For enzymes (e.g., CYP2D6): "potential association in the context of drug metabolism"

COMPUTATIONAL EVIDENCE DESCRIPTION:
- Use "similarity patterns in representation space" instead of "molecular similarity"
- Use "computational similarity analyses" or "similarity-based computational evidence"
- Avoid "embedding similarity" or detailed technical descriptions

EXAMPLE CAUTIOUS PHRASES (Arabic):
- "بناءً على البيانات المتاحة..." (use once at beginning)
- "النموذج الحاسوبي يقترح..." (the computational model suggests)
- "التحليل يشير إلى إمكانية..." (the analysis indicates possibility)
- "قد يكون مرتبطًا وظيفيًا بـ..." (may be functionally associated with)
- "يستند هذا التنبؤ إلى أنماط التشابه الحاسوبي" (this prediction is based on computational similarity patterns)
- "⚠️ تتطلب هذه النتائج الحاسوبية الاستكشافية تحققًا تجريبيًا دقيقًا..." (use once at end)

IMPORTANT ACADEMIC NOTE:
Provide a methodologically sound analysis that clearly distinguishes between computational predictions and established biological knowledge, emphasizing the exploratory and hypothesis-generating nature of these findings.

EXAMPLE TRANSFORMATION:
❌ Before: "Omeprazole increases DHCR24 expression and binds to the gene, modulating sterol pathways."
✅ After: "The computational model predicts an association between Omeprazole and increased DHCR24 expression. The analysis suggests potential involvement in sterol-related pathways. ⚠️ These findings are exploratory and require experimental validation to confirm biological significance."

IMPORTANT RESTRICTION:
You must base your explanation ONLY on the information explicitly provided above.
Do NOT use any external biological knowledge, assumptions, or facts that are not included in the metadata or prediction context.
If information is missing, acknowledge it instead of inventing details.
Your entire explanation must be written in: {llm_lang}.
"""
    
    def try_single_model(model_name):
        """Try a single model with timeout"""
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response.text and len(response.text.strip()) > 50:  # Valid response
                return response.text, model_name
            return None, f"{model_name}: Empty response"
        except Exception as e:
            return None, f"{model_name}: {str(e)[:100]}"
    
    # Try each model in order with timeout
    start_time = time.time()
    total_timeout = 600  # 10 minutes total timeout
    
    for i, model_name in enumerate(AVAILABLE_GEMINI_MODELS):
        # Check if total timeout exceeded
        if time.time() - start_time > total_timeout:
            break
            
        try:
            # Use ThreadPoolExecutor for timeout control
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(try_single_model, model_name)
                try:
                    # 45 second timeout per model (600/14 ≈ 43 seconds per model)
                    result, status = future.result(timeout=45)
                    if result:
                        return result, model_name
                    else:
                        continue
                except TimeoutError:
                    continue
        except Exception as e:
            continue
    
    # If all models failed, return None
    return None, "All models failed"
def get_clean_local_explanation(drug_name, gene_symbol, relation_name, class_prob, drug_classes, bp, lang):
    """Generate a clean local explanation with improved academic framing and varied cautious language"""
    if lang == "العربية":
        return f"""هذا التحليل مبني على نموذج حاسوبي للتنبؤ بالعلاقات الدوائية-الجينية في سياق الأبحاث العصبية.

🔬 **تحليل التفاعل الدوائي-الجيني**

**الدواء:** {drug_name}
**الجين:** {gene_symbol}
**نوع التفاعل المتوقع:** {relation_name}
**مستوى الثقة:** {class_prob:.1%}

**التفسير العلمي:**
بناءً على البيانات المتاحة، النموذج الحاسوبي يقترح وجود ارتباط وظيفي محتمل من نوع {relation_name} بين الدواء {drug_name} والجين {gene_symbol}. التحليل يشير إلى إمكانية أن يكون هذا الارتباط ذا صلة في سياق الأبحاث العصبية.

وفقاً للمعطيات الحالية، الأدوية من هذه الفئة قد تكون مرتبطة وظيفيًا بالتعبير الجيني أو قد تتفاعل مع البروتينات المُشفرة من هذا الجين. يستند هذا التنبؤ إلى أنماط التشابه الحاسوبي في فضاء التمثيل، مما يشير إلى وجود خصائص كامنة مشتركة.

⚠️ تتطلب هذه النتائج الحاسوبية الاستكشافية تحققًا تجريبيًا دقيقًا للتأكد من إمكانياتها العلاجية قبل أي تطبيق سريري."""
    else:
        return f"""This analysis is based on a computational model for predicting drug-gene relationships in neuroscience research context.

🔬 **Drug-Gene Interaction Analysis**

**Drug:** {drug_name}
**Gene:** {gene_symbol}
**Predicted Interaction:** {relation_name}
**Confidence Level:** {class_prob:.1%}

**Scientific Explanation:**
Based on available data, the computational model suggests a potential functional association of type {relation_name} between drug {drug_name} and gene {gene_symbol}. The analysis indicates this association may be relevant in neuroscience research context.

According to current data, drugs of this class may be functionally associated with gene expression changes or may interact with proteins encoded by this gene. This prediction is based on computational similarity patterns in representation space, indicating shared latent characteristics.

⚠️ These exploratory computational predictions require rigorous experimental validation and laboratory research to confirm their therapeutic potential before any clinical application."""

# Display title with proper direction
if selected_lang == "العربية":
    st.markdown(f'<h1 class="arabic-title">{texts["title"]}</h1>', unsafe_allow_html=True)
else:
    st.markdown(f'<h1 class="english-title">{texts["title"]}</h1>', unsafe_allow_html=True)

with st.sidebar.expander(texts["stats_btn"]):
    st.write("Multi‑Class Accuracy: **89.7%** | Macro‑AUC: **94.2%**")

# Display basket counter with proper direction
if selected_lang == "العربية":
    st.sidebar.markdown(f'<h3 class="arabic-sidebar">{texts["reports_added"]} {len(st.session_state.report_list)}</h3>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<h3 class="english-sidebar">{texts["reports_added"]} {len(st.session_state.report_list)}</h3>', unsafe_allow_html=True)

# Display basket contents
if len(st.session_state.report_list) > 0:
    st.sidebar.write("📋 **محتويات السلة:**")
    for i, report in enumerate(st.session_state.report_list):
        with st.sidebar.expander(f"🔬 {report['drug']} ↔ {report['gene']}", expanded=False):
            st.write(f"**العلاقة:** {report['relation']}")
            st.write(f"**الثقة:** {report['prob']}")
            if st.button(f"🗑️ حذف", key=f"delete_{i}"):
                st.session_state.report_list.pop(i)
                st.rerun()
    
    # Show clear button only when basket has items
    if st.sidebar.button(texts["clear_btn"]):
        st.session_state.report_list = []
        st.rerun()
else:
    st.sidebar.info("السلة فارغة")

# Drug selection with proper direction
if selected_lang == "العربية":
    st.markdown(f'<div class="arabic-title">{texts["select_drug"]}</div>', unsafe_allow_html=True)
    selected_name = st.selectbox("", drug_names, label_visibility="collapsed")
else:
    st.markdown(f'<div class="english-title">{texts["select_drug"]}</div>', unsafe_allow_html=True)
    selected_name = st.selectbox("", drug_names, label_visibility="collapsed")
selected_id = drug_map[selected_name]

# ============================================================

c1, c2 = st.columns([1, 1])

with c1:
    if st.button(texts["predict_btn"], use_container_width=True):
        st.session_state.current_explanation = None
        res_list = predict_interaction_with_embeddings(selected_id)

        if res_list:
            st.session_state.all_predictions = res_list
            st.success(f"✔ Found {len(res_list)} predicted relations")
        else:
            st.session_state.all_predictions = None
            st.session_state.current_prediction = None
            st.warning(texts["result_neg"])

if 'all_predictions' in st.session_state and st.session_state.all_predictions:
    preds = st.session_state.all_predictions
    st.subheader(" Relations Summary")
    cols = st.columns(4)
    relation_labels = [
        "NO_LINK",
        "CHEMICALBINDSGENE",
        "CHEMICALINCREASESEXPRESSION",
        "CHEMICALDECREASESEXPRESSION"
    ]
    colors = ["gray", "gray", "gray", "gray"]

    for i, label in enumerate(relation_labels):
        count = sum(1 for r in st.session_state.all_predictions if r["class_name"] == label)
        with cols[i]:
            st.markdown(f"<div style='background-color:{colors[i]}; padding:10px; border-radius:8px; text-align:center;'>"
                        f"<h4 style='color:white;'>{label}</h4>"
                        f"<p style='font-size:16px; color:white;'>{count} genes</p>"
                        f"</div>", unsafe_allow_html=True)
    
    relation_types = [
        "NO_LINK",
        "CHEMICALBINDSGENE",
        "CHEMICALINCREASESEXPRESSION",
        "CHEMICALDECREASESEXPRESSION"
    ]

    # Relation selection with proper direction
    if selected_lang == "العربية":
        st.markdown(f'<div class="arabic-title">{LANG[selected_lang]["select_relation"]}</div>', unsafe_allow_html=True)
        selected_relation = st.selectbox("", relation_types, label_visibility="collapsed")
    else:
        st.markdown(f'<div class="english-title">{LANG[selected_lang]["select_relation"]}</div>', unsafe_allow_html=True)
        selected_relation = st.selectbox("", relation_types, label_visibility="collapsed")
    
    filtered_genes = [
        r for r in st.session_state.all_predictions
        if r["class_name"] == selected_relation
    ]

    if filtered_genes:
        gene_options = [
            f"{gene_id_to_symbol.get(str(r['gene']), 'Gene_' + str(r['gene']))} — Confidence {r['prob']:.1%}"
            for r in filtered_genes
        ]

        # Gene selection with proper direction
        if selected_lang == "العربية":
            st.markdown(f'<div class="arabic-title">{LANG[selected_lang]["select_gene"]}</div>', unsafe_allow_html=True)
            selected_gene_option = st.selectbox("", gene_options, label_visibility="collapsed")
        else:
            st.markdown(f'<div class="english-title">{LANG[selected_lang]["select_gene"]}</div>', unsafe_allow_html=True)
            selected_gene_option = st.selectbox("", gene_options, label_visibility="collapsed")
        
        selected_idx = gene_options.index(selected_gene_option)
        st.session_state.current_prediction = filtered_genes[selected_idx]

        p = st.session_state.current_prediction
        st.subheader(" Relation Class Probabilities")

        probs = p['all_probs']
        labels = [
            "NO_LINK",
            "CHEMICALBINDSGENE",
            "CHEMICALINCREASESEXPRESSION",
            "CHEMICALDECREASESEXPRESSION"
        ]
        colors = ["gray", "gray", "gray", "gray"]

        for i in range(4):
            st.markdown(f"<div style='background-color:{colors[i]}; padding:8px; border-radius:6px;'>"
                        f"<strong style='color:white;'>{labels[i]}</strong>: "
                        f"<span style='color:white;'>{probs[i]*100:.1f}%</span>"
                        f"</div>", unsafe_allow_html=True)
    else:
        st.warning("لا توجد جينات لهذا النوع من العلاقة.")

# ============================================================

with c2:
    if st.session_state.get("current_prediction") and st.button(texts["explain_btn"], use_container_width=True):
        with st.spinner("Gemini AI is analyzing the selected gene..."):
            p = st.session_state.current_prediction
            gene_symbol = gene_id_to_symbol.get(str(p['gene']), f"Gene_{p['gene']}")
            
            # Try all Gemini models first
            ai_explanation, used_model = try_all_gemini_models(
                drug_name=selected_name,
                drug_id=selected_id,
                gene_name=gene_symbol,
                gene_id=p['gene'],
                class_id=p['class_id'],
                class_prob=p['prob'],
                drug_emb=p['drug_emb'],
                gene_emb=p['gene_emb'],
                drug_metadata=drug_metadata,
                gene_metadata=gene_metadata,
                lang=selected_lang
            )
            
            if ai_explanation:
                # AI explanation worked
                st.session_state.current_explanation = ai_explanation
                st.session_state.used_model = used_model
                st.sidebar.success(f"✅ تم التفسير بواسطة: {used_model.split('/')[-1]}")
            else:
                # All AI models failed, use local explanation
                st.session_state.current_explanation = get_clean_local_explanation(
                    selected_name,
                    gene_symbol,
                    p['class_name'],
                    p['prob'],
                    "No known drug classes",
                    "No known biological processes",
                    selected_lang
                )
                st.session_state.used_model = "Local Explanation"
                st.sidebar.warning("⚠️ تم استخدام التفسير المحلي")

# ============================================================

if st.session_state.get("current_explanation"):
    explanation_text = st.session_state.current_explanation
    
    # Add header based on explanation type
    if st.session_state.get('used_model') == "Local Explanation":
        header = "🏠 Local Interpretation:"
    else:
        header = "🤖 AI Generated:"
    
    # Display content with proper direction
    if selected_lang == "العربية":
        st.markdown(f'<div class="ai-header">{header}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="direction: rtl; text-align: right; background-color: #e1f5fe; padding: 15px; border-radius: 8px; border-left: 4px solid #0288d1;">{explanation_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-header">{header}</div>', unsafe_allow_html=True)
        st.info(explanation_text)

    if st.button(texts["add_report"], use_container_width=True):
        p = st.session_state.current_prediction
        gene_symbol = gene_id_to_symbol.get(str(p['gene']), f"Gene_{p['gene']}")
        st.session_state.report_list.append({
            "drug": selected_name,
            "gene": gene_symbol,  # Use gene symbol instead of ID
            "gene_id": p['gene'],  # Keep ID for reference
            "relation": p['class_name'],
            "prob": f"{p['prob']:.1%}",
            "explanation": st.session_state.current_explanation
        })
        st.toast(f"Added {gene_symbol} to report basket!")

# Display basket contents in main area if not empty
if len(st.session_state.report_list) > 0:
    st.write("---")
    st.subheader(f"🛒 سلة التقارير ({len(st.session_state.report_list)} تحليل)")
    
    for i, report in enumerate(st.session_state.report_list):
        with st.expander(f"📊 تحليل {i+1}: {report['drug']} ↔ {report['gene']}", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**الدواء:** {report['drug']}")
                st.write(f"**الجين:** {report['gene']}")
                st.write(f"**نوع العلاقة:** {report['relation']}")
                st.write(f"**مستوى الثقة:** {report['prob']}")
            
            with col2:
                if st.button(f"🗑️ حذف التحليل", key=f"main_delete_{i}"):
                    st.session_state.report_list.pop(i)
                    st.rerun()
            
            with col3:
                if st.button(f"📋 عرض التفسير", key=f"show_explanation_{i}"):
                    st.info(report['explanation'])
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 تصدير كـ PDF", use_container_width=True):
            try:
                pdf_buffer = generate_pdf_report(st.session_state.report_list, selected_lang)
                
                # Create download button
                st.download_button(
                    label="📥 تحميل ملف PDF",
                    data=pdf_buffer,
                    file_name=f"alzheimer_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("✅ تم إنشاء ملف PDF بنجاح!")
            except Exception as e:
                st.error(f"❌ خطأ في إنشاء PDF: {str(e)}")
    with col2:
        if st.button("🗑️ تفريغ السلة", use_container_width=True):
            st.session_state.report_list = []
            st.rerun()

st.write("---")
st.caption(texts["status_ready"])