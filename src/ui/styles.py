"""
SentinelFetal Professional UI Styles.

This module provides custom CSS styling for a professional medical dashboard look.
All styles are designed to hide Streamlit's default chrome and create a clean,
medical-grade interface.

Color Palette (Medical):
    - Primary: #1E3A5F (Navy) - Headers, sidebar
    - Background: #F8FAFC (Light gray) - Main background
    - White: #FFFFFF - Cards, panels
    - Category 1 (Normal): #22C55E (Green)
    - Category 2 (Intermediate): #F59E0B (Amber)
    - Category 3 (Pathological): #EF4444 (Red)
    - Accent: #3B82F6 (Blue) - Buttons, links

Typography:
    - Primary: Inter, -apple-system, sans-serif
    - Monospace: SF Mono, Menlo (for metrics)

Usage:
    >>> from src.ui.styles import inject_custom_css, COLORS
    >>> inject_custom_css()  # Call at top of main()
"""

import streamlit as st

# ============================================================================
# Color Constants
# ============================================================================

COLORS = {
    # Primary palette
    'primary': '#1E3A5F',
    'primary_light': '#2D4A6F',
    'primary_dark': '#0F2A4F',
    
    # Background
    'background': '#F8FAFC',
    'background_dark': '#E2E8F0',
    'card': '#FFFFFF',
    
    # Text
    'text_primary': '#1E293B',
    'text_secondary': '#64748B',
    'text_muted': '#94A3B8',
    
    # Category colors
    'category_1': '#22C55E',  # Green - Normal
    'category_1_light': '#DCFCE7',
    'category_2': '#F59E0B',  # Amber - Intermediate
    'category_2_light': '#FEF3C7',
    'category_3': '#EF4444',  # Red - Pathological
    'category_3_light': '#FEE2E2',
    
    # Accent
    'accent': '#3B82F6',
    'accent_light': '#DBEAFE',
    
    # Borders
    'border': '#E2E8F0',
    'border_dark': '#CBD5E1',
}


# ============================================================================
# CSS Styles
# ============================================================================

def get_hide_streamlit_chrome_css() -> str:
    """CSS to hide Streamlit's default header, footer, and menu."""
    return """
        /* Hide Streamlit header */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        
        /* Hide hamburger menu */
        #MainMenu {
            display: none !important;
        }
        
        /* Hide footer */
        footer {
            display: none !important;
        }
        
        /* Hide "Deploy" button */
        .stDeployButton {
            display: none !important;
        }
        
        /* Hide "Made with Streamlit" */
        .viewerBadge_container__1QSob {
            display: none !important;
        }
        
        /* Remove top padding from main content */
        .block-container {
            padding-top: 1rem !important;
        }
    """


def get_sidebar_css() -> str:
    """CSS for professional sidebar styling."""
    return f"""
        /* Sidebar background */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
        }}
        
        /* Sidebar content */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            color: white !important;
        }}
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: white !important;
        }}
        
        [data-testid="stSidebar"] label {{
            color: rgba(255, 255, 255, 0.9) !important;
        }}
        
        [data-testid="stSidebar"] .stSelectbox label {{
            color: rgba(255, 255, 255, 0.9) !important;
        }}
        
        /* Sidebar dividers */
        [data-testid="stSidebar"] hr {{
            border-color: rgba(255, 255, 255, 0.2);
        }}
        
        /* Sidebar caption */
        [data-testid="stSidebar"] .stCaption {{
            color: rgba(255, 255, 255, 0.6) !important;
        }}
        
        /* Sidebar info box */
        [data-testid="stSidebar"] .stAlert {{
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
        }}
    """


def get_main_content_css() -> str:
    """CSS for main content area styling."""
    return f"""
        /* Main background */
        .stApp {{
            background-color: {COLORS['background']};
        }}
        
        /* Main content area */
        .main .block-container {{
            max-width: 1400px;
            padding: 1rem 2rem;
        }}
        
        /* Card-like containers */
        .stExpander {{
            background-color: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}
        
        /* Metric cards */
        [data-testid="stMetric"] {{
            background-color: {COLORS['card']};
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid {COLORS['border']};
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }}
        
        [data-testid="stMetricLabel"] {{
            color: {COLORS['text_secondary']} !important;
            font-size: 0.85rem !important;
        }}
        
        [data-testid="stMetricValue"] {{
            color: {COLORS['text_primary']} !important;
            font-weight: 600 !important;
        }}
    """


def get_typography_css() -> str:
    """CSS for typography styling."""
    return f"""
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Base font */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        
        /* Headings */
        h1 {{
            color: {COLORS['primary']} !important;
            font-weight: 700 !important;
            font-size: 1.75rem !important;
        }}
        
        h2 {{
            color: {COLORS['text_primary']} !important;
            font-weight: 600 !important;
            font-size: 1.25rem !important;
        }}
        
        h3 {{
            color: {COLORS['text_primary']} !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }}
        
        /* Body text */
        p, li {{
            color: {COLORS['text_primary']};
            line-height: 1.6;
        }}
        
        /* Subheaders */
        .stSubheader {{
            color: {COLORS['text_primary']} !important;
            font-weight: 600 !important;
            border-bottom: 2px solid {COLORS['accent']};
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }}
    """


def get_alert_css() -> str:
    """CSS for custom alert/category banners."""
    return f"""
        /* Category 1 (Normal) banner */
        .category-banner-1 {{
            background: linear-gradient(135deg, {COLORS['category_1']} 0%, #16A34A 100%);
            padding: 1.25rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(34, 197, 94, 0.2);
        }}
        
        /* Category 2 (Intermediate) banner */
        .category-banner-2 {{
            background: linear-gradient(135deg, {COLORS['category_2']} 0%, #D97706 100%);
            padding: 1.25rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(245, 158, 11, 0.2);
        }}
        
        /* Category 3 (Pathological) banner */
        .category-banner-3 {{
            background: linear-gradient(135deg, {COLORS['category_3']} 0%, #DC2626 100%);
            padding: 1.25rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(239, 68, 68, 0.2);
        }}
        
        .category-banner-1 h2,
        .category-banner-2 h2,
        .category-banner-3 h2 {{
            color: white !important;
            margin: 0 !important;
            font-size: 1.5rem !important;
        }}
        
        .category-banner-1 p,
        .category-banner-2 p,
        .category-banner-3 p {{
            color: rgba(255, 255, 255, 0.9) !important;
            margin: 0.5rem 0 0 0 !important;
        }}
        
        /* Finding cards */
        .finding-card {{
            background-color: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }}
        
        .finding-card:hover {{
            border-color: {COLORS['accent']};
            box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
        }}
        
        /* Recommendation styling based on category */
        .recommendation-normal {{
            background-color: {COLORS['category_1_light']};
            border-left: 4px solid {COLORS['category_1']};
            padding: 0.75rem 1rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.5rem;
        }}
        
        .recommendation-warning {{
            background-color: {COLORS['category_2_light']};
            border-left: 4px solid {COLORS['category_2']};
            padding: 0.75rem 1rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.5rem;
        }}
        
        .recommendation-critical {{
            background-color: {COLORS['category_3_light']};
            border-left: 4px solid {COLORS['category_3']};
            padding: 0.75rem 1rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.5rem;
        }}
    """


def get_button_css() -> str:
    """CSS for button styling."""
    return f"""
        /* Primary buttons */
        .stButton > button {{
            background-color: {COLORS['accent']};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .stButton > button:hover {{
            background-color: #2563EB;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
            transform: translateY(-1px);
        }}
        
        /* Download button */
        .stDownloadButton > button {{
            background-color: {COLORS['primary']};
            color: white;
        }}
    """


def get_plotly_css() -> str:
    """CSS for Plotly chart containers."""
    return f"""
        /* Plotly chart container */
        .stPlotlyChart {{
            background-color: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}
    """


def get_progress_css() -> str:
    """CSS for progress indicators."""
    return f"""
        /* Progress bar */
        .stProgress > div > div {{
            background-color: {COLORS['accent']};
        }}
        
        /* Status container */
        .stStatus {{
            border-radius: 8px;
        }}
    """


def get_full_css() -> str:
    """Get all CSS combined."""
    return f"""
    <style>
        {get_hide_streamlit_chrome_css()}
        {get_sidebar_css()}
        {get_main_content_css()}
        {get_typography_css()}
        {get_alert_css()}
        {get_button_css()}
        {get_plotly_css()}
        {get_progress_css()}
    </style>
    """


def inject_custom_css():
    """
    Inject custom CSS into the Streamlit app.
    
    Call this at the top of main() function.
    
    Example:
        >>> def main():
        ...     inject_custom_css()
        ...     st.title("My App")
    """
    st.markdown(get_full_css(), unsafe_allow_html=True)


# ============================================================================
# HTML Component Helpers
# ============================================================================

def category_banner_html(category: int, headline: str, record_id: str, confidence: float) -> str:
    """
    Generate HTML for category banner.
    
    Args:
        category: Category number (1, 2, or 3)
        headline: Alert headline
        record_id: Patient/record identifier
        confidence: Confidence score (0-1)
        
    Returns:
        HTML string for the banner
    """
    return f"""
        <div class="category-banner-{category}">
            <h2>{headline}</h2>
            <p>רשומה: {record_id} | ביטחון: {confidence:.1%}</p>
        </div>
    """


def finding_card_html(finding: str, icon: str = "•") -> str:
    """
    Generate HTML for a finding card.
    
    Args:
        finding: Finding text
        icon: Icon to display (default: bullet)
        
    Returns:
        HTML string for the card
    """
    return f"""
        <div class="finding-card">
            <span>{icon}</span> {finding}
        </div>
    """


def recommendation_html(recommendation: str, category: int) -> str:
    """
    Generate HTML for a recommendation.
    
    Args:
        recommendation: Recommendation text
        category: Category number for styling
        
    Returns:
        HTML string for the recommendation
    """
    style_class = {
        1: "recommendation-normal",
        2: "recommendation-warning",
        3: "recommendation-critical"
    }.get(category, "recommendation-normal")
    
    return f"""
        <div class="{style_class}">
            • {recommendation}
        </div>
    """


def section_header_html(title: str, icon: str = "") -> str:
    """
    Generate HTML for a section header.
    
    Args:
        title: Section title
        icon: Emoji icon (optional)
        
    Returns:
        HTML string for the header
    """
    return f"""
        <h3 style="
            color: {COLORS['text_primary']};
            font-weight: 600;
            border-bottom: 2px solid {COLORS['accent']};
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        ">
            {icon} {title}
        </h3>
    """


# ============================================================================
# Utility Functions
# ============================================================================

def get_category_gradient(category: int) -> str:
    """Get CSS gradient for a category."""
    gradients = {
        1: f"linear-gradient(135deg, {COLORS['category_1']} 0%, #16A34A 100%)",
        2: f"linear-gradient(135deg, {COLORS['category_2']} 0%, #D97706 100%)",
        3: f"linear-gradient(135deg, {COLORS['category_3']} 0%, #DC2626 100%)"
    }
    return gradients.get(category, gradients[1])


def get_category_bg_color(category: int) -> str:
    """Get background color for a category."""
    colors = {
        1: COLORS['category_1_light'],
        2: COLORS['category_2_light'],
        3: COLORS['category_3_light']
    }
    return colors.get(category, colors[1])
