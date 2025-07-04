# streamlit library
import streamlit as st
import pickle
import sys
import pandas as pd
import numpy as np
from kedro.io import DataCatalog

st.set_page_config(layout="wide")
st.title('Data Catalog from Kedro')

#----------------------------------------------------------------------------
# KEDRO CONFIG

from pathlib import Path
from kedro.framework.project import configure_project

package_name = "wine_project"  # Use fixed package name as Path(__file__).parent.name may return "pages"
configure_project(package_name)

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

# Fix config path to point to project root conf directory
conf_path = str(Path(__file__).resolve().parent.parent.parent / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
catalog = conf_loader["catalog"]

# Define dataset lists
available_datasets = [
    'wine_raw_data',
    'wine_ingested_data',
    'batch_preprocessed',
    'train_preprocessed',
    'train_all_features',
    'batch_feature_engineered',
    'train_feature_engineered',
    'batch_feature_engineered_one_hot',
    'train_feature_engineered_one_hot',
    'X_train',
    'X_val',
    'y_train',
    'y_val',
    'X_train_one_hot',
    'X_val_one_hot',
    'y_train_one_hot',
    'y_val_one_hot'
]

# Filter to only include datasets actually in the catalog
valid_datasets = [ds for ds in available_datasets if ds in catalog]

# Create sidebar for dataset selection
choice = st.radio('**Available Dataset:**', valid_datasets, index=0)

# Load the data catalog
datacatalog = DataCatalog.from_config(catalog)

# Try to load the selected dataset
try:
    with st.spinner(f"Loading {choice}..."):
        dataset = datacatalog.load(choice)
    
    # Check if it's a pandas DataFrame
    if isinstance(dataset, pd.DataFrame):
        # Show basic dataset info
        st.write(f"## {choice}")
        st.write(f"**Shape:** {dataset.shape[0]} rows × {dataset.shape[1]} columns")
        
        # Display the first few rows of the dataset
        st.write("### Preview")
        st.dataframe(dataset.head(10))
        
        # Show summary statistics
        with st.expander("Summary Statistics"):
            # Get numeric columns
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write("#### Numeric Columns")
                st.dataframe(dataset[numeric_cols].describe())
            
            # Get non-numeric columns
            cat_cols = dataset.select_dtypes(exclude=[np.number]).columns.tolist()
            if cat_cols:
                st.write("#### Categorical Columns")
                for col in cat_cols[:5]:  # Limit to first 5 columns to avoid clutter
                    st.write(f"**{col}**")
                    st.write(dataset[col].value_counts().head(10))
        
        # Missing values analysis
        with st.expander("Missing Values Analysis"):
            missing = dataset.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if not missing.empty:
                st.write("#### Columns with Missing Values")
                missing_df = pd.DataFrame({
                    'Column': missing.index,
                    'Missing Count': missing.values,
                    'Missing Percentage': (missing.values / len(dataset) * 100).round(2)
                })
                st.dataframe(missing_df)
            else:
                st.write("No missing values found!")
          # Options for profile report generation
        st.write("### Data Profiling")
        profile_type = st.radio(
            "Select profile report type:",
            ["Basic (faster, less visuals)", "Complete (slower, more visuals)"],
            index=0
        )
        
        if st.button("Generate Profile Report"):
            try:
                from ydata_profiling import ProfileReport
                
                # Create progress bar
                progress_text = "Generating profile report. This may take a while..."
                progress_bar = st.progress(0, text=progress_text)
                
                # Configure based on user selection
                minimal_mode = profile_type == "Basic (faster, less visuals)"
                
                # Set config based on profile type
                if minimal_mode:
                    # Create the profile with minimal=True for faster processing
                    profile = ProfileReport(
                        dataset, 
                        title=f"{choice} Profile Report",
                        minimal=True,  # Faster processing
                        progress_bar=False,  # Disable internal progress bar
                        correlations=None  # Skip correlations for speed
                    )
                    progress_bar.progress(25, text="Processing basic statistics...")
                else:
                    # Create full profile with more visualizations
                    profile = ProfileReport(
                        dataset, 
                        title=f"{choice} Profile Report",
                        minimal=False,  # Full processing
                        progress_bar=False,
                        correlations={
                            "auto": {"calculate": True},
                            "pearson": {"calculate": True},
                            "spearman": {"calculate": True},
                        },
                        plot={
                            "histogram": {"bayesian_blocks_bins": True},
                            "correlation": {
                                "cmap": "RdBu_r",
                                "bad": "#000000"
                            },
                            "missing": {"force_labels": True},
                        },
                        explorative=True
                    )
                    progress_bar.progress(25, text="Processing statistics and generating visualizations...")
                
                # Update progress
                progress_bar.progress(75, text="Creating HTML report...")
                
                # Generate HTML report
                html_report = profile.to_html()
                
                # Update progress
                progress_bar.progress(100, text="Done!")
                
                # Display the report in an iframe with increased height
                st.components.v1.html(html_report, height=800, scrolling=True)
                
            except Exception as e:
                st.error(f"Error generating profile report: {str(e)}")
                st.info("Try using the 'Basic' profile type for this dataset, or select a smaller dataset.")
                
        # Additional visualization options    
        st.write("### Custom Visualizations")
        
        # Only show visualization options if we have a DataFrame with numeric columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            # Plot correlation heatmap
            if st.checkbox("Show Correlation Heatmap"):
                st.write("#### Correlation Heatmap")
                corr = dataset[numeric_cols].corr()
                
                # Use Streamlit's native charting
                st.write("This shows how numeric variables relate to each other:")
                st.dataframe(corr.style.background_gradient(cmap='coolwarm'), use_container_width=True)
            
            # Add categorical correlation using Cramer's V
            cat_cols = dataset.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            if len(cat_cols) >= 2 and st.checkbox("Show Categorical Correlation Heatmap"):
                st.write("#### Categorical Correlation Heatmap (using Cramér's V)")
                st.write("This shows how categorical variables relate to each other (values close to 1 indicate strong association):")
                
                with st.spinner("Calculating categorical correlations (this may take a while for large datasets)..."):
                    # Define Cramer's V function
                    def cramers_v(x, y):
                        """Calculate Cramér's V correlation between two categorical variables."""
                        try:
                            confusion_matrix = pd.crosstab(x, y)
                            chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
                            n = confusion_matrix.sum().sum()
                            phi2 = chi2 / n
                            r, k = confusion_matrix.shape
                            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
                            rcorr = r - ((r - 1) ** 2) / (n - 1)
                            kcorr = k - ((k - 1) ** 2) / (n - 1)
                            
                            # Handle division by zero
                            denominator = min((kcorr - 1), (rcorr - 1))
                            if denominator <= 0:
                                return 0.0
                                
                            return np.sqrt(phi2corr / denominator)
                        except Exception as e:
                            st.warning(f"Error calculating Cramér's V between {x.name} and {y.name}: {str(e)}")
                            return np.nan
                    
                    try:
                        # Import scipy for chi2_contingency
                        import scipy.stats
                        import seaborn as sns
                        import matplotlib.pyplot as plt
                        
                        # Generate correlation matrix
                        cat_corr_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)
                        
                        # Calculate Cramér's V for each pair of categorical variables
                        # Use only the first 1000 rows for performance if dataset is large
                        sample_data = dataset.head(1000) if len(dataset) > 1000 else dataset
                        
                        for i, col1 in enumerate(cat_cols):
                            for j, col2 in enumerate(cat_cols):
                                if i == j:
                                    cat_corr_matrix.loc[col1, col2] = 1.0
                                else:
                                    cat_corr_matrix.loc[col1, col2] = cramers_v(
                                        sample_data[col1], sample_data[col2]
                                    )
                        
                        # Convert the matrix to float type to avoid the "Image data of dtype object" error
                        cat_corr_matrix = cat_corr_matrix.astype(float)
                        
                        # Handle any remaining NaN values
                        cat_corr_matrix = cat_corr_matrix.fillna(0)
                        
                        # Round values to 1 decimal place for cleaner display
                        cat_corr_matrix = cat_corr_matrix.round(2)
                        
                        # Display as a heatmap with matplotlib and seaborn
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Create a mask for the upper triangle
                        mask = np.triu(np.ones_like(cat_corr_matrix, dtype=bool))
                        
                        # Create the heatmap with a red-to-blue color scheme
                        sns.heatmap(
                            cat_corr_matrix, 
                            mask=mask,
                            cmap='coolwarm_r',  # Red for high correlation, blue for low
                            vmin=0, 
                            vmax=1,
                            annot=True,  # Show values
                            fmt=".2f",   # Format as 2
                            linewidths=0.5,
                            ax=ax,
                            square=True,
                            cbar_kws={"shrink": 0.8, "label": "Cramér's V Correlation"}
                        )
                        
                        plt.title("Categorical Feature Correlation (Cramér's V)", fontsize=14)
                        plt.tight_layout()
                        
                        # Display the plot in Streamlit
                        st.pyplot(fig)
                        
                        # Also display the raw matrix as a dataframe for reference
                        with st.expander("Show correlation values as table"):
                            st.dataframe(
                                cat_corr_matrix.style.background_gradient(cmap='coolwarm_r', vmin=0, vmax=1),
                                use_container_width=True
                            )
                        
                        st.info("Note: Values close to 0 indicate no association, while values close to 1 indicate strong association. Only the lower triangle is shown to avoid redundancy.")
                        
                        if len(dataset) > 1000:
                            st.caption("For performance reasons, this analysis used only the first 1000 rows of the dataset.")
                    
                    except Exception as e:
                        st.error(f"Error creating categorical correlation matrix: {str(e)}")
                        st.info("This could be due to large categorical columns with many unique values. Try with a smaller dataset.")
                
            # Scatter plot between two columns
            if st.checkbox("Show Scatter Plot"):
                st.write("#### Scatter Plot")
                col1, col2 = st.columns(2)
                with col1:
                    x_column = st.selectbox("X-axis", numeric_cols)
                with col2:
                    y_column = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                    
                try:
                    chart_data = pd.DataFrame({
                        x_column: dataset[x_column],
                        y_column: dataset[y_column]
                    })
                    st.scatter_chart(chart_data, x=x_column, y=y_column)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")
    else:
        # Handle non-DataFrame objects
        st.warning(f"The dataset '{choice}' is not a pandas DataFrame.")
        st.write(f"Type: {type(dataset)}")
        st.write("Preview:")
        st.write(dataset)
        
except Exception as e:
    st.error(f"Error loading dataset '{choice}': {str(e)}")
    st.write("Please check if the dataset exists and is properly configured in your catalog.")