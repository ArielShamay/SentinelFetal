# -*- coding: utf-8 -*-
"""
SentinelFetal Professional UI Styles - Clinical Minimalism.

This module provides custom CSS styling for a clean, clinical dashboard.
Design principles:
    - White background (#FFFFFF) - Clean, medical-grade
    - Black text (#000000) - Maximum readability
    - Minimal decoration - No gradients or large banners
    - Semantic colors only for status: Green/Orange/Red

Color Palette (Clinical Minimalism):
    - Background: #FFFFFF (Pure white)
    - Text: #000000 (Pure black)
    - Category 1 (Normal): #28a745 (Green)
    - Category 2 (Intermediate): #fd7e14 (Orange)
    - Category 3 (Pathological): #dc3545 (Red)
    - Border: #E5E5E5 (Light gray)

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
    # Clinical Minimalism Palette
    # Background - Pure white
    'background': '#FFFFFF',
    'card': '#FFFFFF',

    # Text - Pure black for maximum readability
    'text_primary': '#000000',
    'text_secondary': "#2F2C2C",
    'text_muted': '#666666',

    # Category colors (semantic only)
    'category_1': '#28a745',  # Green - Normal
    'category_1_light': '#d4edda',
    'category_2': '#fd7e14',  # Orange - Intermediate
    'category_2_light': '#fff3cd',
    'category_3': '#dc3545',  # Red - Pathological
    'category_3_light': '#f8d7da',

    # Accent - Red instead of black
    'accent': '#DC3545',
    'accent_light': '#f5f5f5',

    # Borders - Light gray
    'border': '#E5E5E5',
    'border_dark': '#CCCCCC',

    # Legacy compatibility (some code may reference these) - Red
    'primary': '#DC3545',
    'primary_light': '#333333',
    'primary_dark': '#DC3545',
    'background_dark': '#F5F5F5',
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
    """CSS for sidebar styling - Clinical Minimalism (collapsed by default)."""
    return f"""
        /* Sidebar background - white/light gray */
        [data-testid="stSidebar"] {{
            background-color: {COLORS['accent_light']};
            border-right: 1px solid {COLORS['border']};
        }}

        /* Sidebar content - black text */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            color: {COLORS['text_primary']} !important;
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: {COLORS['text_primary']} !important;
        }}

        [data-testid="stSidebar"] label {{
            color: {COLORS['text_primary']} !important;
        }}

        [data-testid="stSidebar"] .stSelectbox label {{
            color: {COLORS['text_primary']} !important;
        }}

        /* Sidebar dividers */
        [data-testid="stSidebar"] hr {{
            border-color: {COLORS['border']};
        }}

        /* Sidebar caption */
        [data-testid="stSidebar"] .stCaption {{
            color: {COLORS['text_muted']} !important;
        }}
    """


def get_main_content_css() -> str:
    """CSS for main content area styling - Clinical Minimalism."""
    return f"""
        /* Main background - Pure white */
        .stApp {{
            background-color: {COLORS['background']};
        }}

        /* Main content area */
        .main .block-container {{
            max-width: 1600px;
            padding: 1rem 1.5rem;
            background-color: {COLORS['background']};
        }}

        /* Card-like containers - minimal styling */
        .stExpander {{
            background-color: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
        }}

        /* Metric cards - clean design */
        [data-testid="stMetric"] {{
            background-color: {COLORS['card']};
            padding: 0.75rem;
            border-radius: 4px;
            border: 1px solid {COLORS['border']};
        }}

        [data-testid="stMetricLabel"] {{
            color: {COLORS['text_primary']} !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
        }}

        [data-testid="stMetricValue"] {{
            color: {COLORS['text_primary']} !important;
            font-weight: 600 !important;
        }}

        /* Remove shadows for cleaner look */
        .stExpander, [data-testid="stMetric"] {{
            box-shadow: none !important;
        }}

        /* Form controls - force white background with black text */
        [data-testid="stSelectbox"] > div > div {
            background: #FFFFFF !important;
            color: #000000 !important;
            border: 1px solid {COLORS['border']};
        }

        [data-testid="stSelectbox"] [role="combobox"],
        [data-testid="stSelectbox"] [data-baseweb="select"] {
            background: #FFFFFF !important;
            color: #000000 !important;
        }

        /* Dropdown menu */
        [role="listbox"],
        [data-baseweb="menu"] {
            background: #FFFFFF !important;
            color: #000000 !important;
            border: 1px solid {COLORS['border']};
        }

        /* Dropdown options */
        [role="option"] {
            color: #000000 !important;
            background: #FFFFFF !important;
        }

        /* Number inputs */
        [data-testid="stNumberInput"] input {
            background: #FFFFFF !important;
            color: #000000 !important;
            border: 1px solid {COLORS['border']};
        }

        /* Expander headers */
        [data-testid="stExpander"] summary {
            background: #FFFFFF !important;
            color: #000000 !important;
            border-bottom: 1px solid {COLORS['border']};
        }
    """


def get_typography_css() -> str:
    """CSS for typography styling - Clinical Minimalism (black text)."""
    return f"""
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Base font - black on white */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: {COLORS['text_primary']};
        }}

        /* All headings - pure black */
        h1, h2, h3, h4, h5, h6 {{
            color: {COLORS['text_primary']} !important;
        }}

        h1 {{
            font-weight: 700 !important;
            font-size: 1.75rem !important;
        }}

        h2 {{
            font-weight: 600 !important;
            font-size: 1.25rem !important;
        }}

        h3 {{
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }}

        /* Body text - black */
        p, li, span, div {{
            color: {COLORS['text_primary']};
        }}

        /* Links - black with underline */
        a {{
            color: {COLORS['text_primary']} !important;
            text-decoration: underline;
        }}

        /* Subheaders - minimal styling */
        .stSubheader {{
            color: {COLORS['text_primary']} !important;
            font-weight: 600 !important;
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }}
    """


def get_alert_css() -> str:
    """CSS for category indicators - Clinical Minimalism (no large banners)."""
    return f"""
        /* Category indicators - small, inline colored dots/text */
        .category-indicator {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .category-indicator-1 {{
            color: {COLORS['category_1']};
            border: 1px solid {COLORS['category_1']};
        }}

        .category-indicator-2 {{
            color: {COLORS['category_2']};
            border: 1px solid {COLORS['category_2']};
        }}

        .category-indicator-3 {{
            color: {COLORS['category_3']};
            border: 1px solid {COLORS['category_3']};
        }}

        /* Status dot - minimal colored indicator */
        .status-dot {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }}

        .status-dot-1 {{ background-color: {COLORS['category_1']}; }}
        .status-dot-2 {{ background-color: {COLORS['category_2']}; }}
        .status-dot-3 {{ background-color: {COLORS['category_3']}; }}

        /* Category banners - ensure white background + black text */
        .category-banner-1, .category-banner-2, .category-banner-3 {{
            background: #FFFFFF;
            color: #000000;
            border: 1px solid {COLORS['border']};
            border-left-width: 6px;
            padding: 0.85rem 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }}

        .category-banner-1 {{ border-left-color: {COLORS['category_1']}; }}
        .category-banner-2 {{ border-left-color: {COLORS['category_2']}; }}
        .category-banner-3 {{ border-left-color: {COLORS['category_3']}; }}

        .category-banner-1 h2, .category-banner-2 h2, .category-banner-3 h2,
        .category-banner-1 p,  .category-banner-2 p,  .category-banner-3 p {{
            margin: 0;
            color: #000000;
        }}

        /* Finding cards - clean, minimal */
        .finding-card {{
            background-color: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
        }}

        .finding-card:hover {{
            border-color: {COLORS['border_dark']};
        }}

        /* Recommendation styling - subtle colored border */
        .recommendation-normal {{
            border-left: 3px solid {COLORS['category_1']};
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.5rem;
        }}

        .recommendation-warning {{
            border-left: 3px solid {COLORS['category_2']};
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.5rem;
        }}

        .recommendation-critical {{
            border-left: 3px solid {COLORS['category_3']};
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.5rem;
        }}

        /* Hide large st.error/warning/success banners - use colored text instead */
        .stAlert {{
            background-color: #FFFFFF !important;
            border: 1px solid {COLORS['border']} !important;
            padding: 0.75rem 1rem !important;
            border-radius: 4px !important;
            box-shadow: none !important;
        }}

        /* Category colored text for alerts */
        .stAlert [data-testid="stAlert"] {{
            color: inherit !important;
        }}

        /* Specific alert colors */
        .stAlert-success {{
            color: {COLORS['category_1']} !important;
        }}

        .stAlert-warning {{
            color: {COLORS['category_2']} !important;
        }}

        .stAlert-error {{
            color: {COLORS['category_3']} !important;
        }}

        .stAlert-info {{
            color: {COLORS['text_primary']} !important;
        }}
    """


def get_button_css() -> str:
    """CSS for button styling - Clinical Minimalism (black/white)."""
    return f"""
        /* Primary buttons - red on white */
        .stButton > button {{
            background-color: {COLORS['accent']};
            color: {COLORS['background']};
            border: 1px solid {COLORS['accent']};
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.15s ease;
        }}

        .stButton > button:hover {{
            background-color: {COLORS['background']};
            color: {COLORS['accent']};
        }}

        /* Download button */
        .stDownloadButton > button {{
            background-color: {COLORS['accent']};
            color: {COLORS['background']};
        }}

        /* Start/Stop buttons - functional colors */
        .start-btn {{
            background-color: {COLORS['category_1']} !important;
            border-color: {COLORS['category_1']} !important;
        }}

        .stop-btn {{
            background-color: {COLORS['category_3']} !important;
            border-color: {COLORS['category_3']} !important;
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
