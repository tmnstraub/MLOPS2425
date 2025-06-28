"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 0.19.5
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

def visualize_data_unit_test_results(df_validation):
    """
    Create visualizations from the data unit test results
    """
    logger.info("Creating data unit test visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs("data/08_reporting", exist_ok=True)
    
    # Save the raw data
    df_validation.to_csv("data/08_reporting/data_unit_test_results.csv", index=False)
    logger.info("Saved raw test results to data/08_reporting/data_unit_test_results.csv")
    
    # Check if Test Description column exists, if not create it
    if 'Test Description' not in df_validation.columns:
        # Define mapping from expectation types to readable descriptions
        description_map = {
            "expect_table_column_count_to_equal": "Number of features in dataset",
            "expect_column_values_to_be_between": "Value range check",
            "expect_column_values_to_be_of_type": "Data type check",
            "expect_table_row_count_to_equal_other_table": "No duplicate records check"
        }
        
        # Add the column with mapped descriptions
        df_validation['Test Description'] = df_validation['Expectation Type'].map(description_map)
        
        # Add more context based on column and parameters
        for idx, row in df_validation.iterrows():
            if not pd.isna(row['Column']) and row['Column']:
                df_validation.at[idx, 'Test Description'] += f" ({row['Column']})"
            
            if row['Expectation Type'] == "expect_column_values_to_be_between":
                min_val = row['Min Value'] if not pd.isna(row['Min Value']) else "any"
                max_val = row['Max Value'] if not pd.isna(row['Max Value']) else "any"
                df_validation.at[idx, 'Test Description'] += f" [{min_val} to {max_val}]"
    
    # Plot 1: Success by test type (using Test Description)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    
    # Always use Test Description instead of Expectation Type
    success_by_type = df_validation.groupby('Test Description')['Success'].mean().sort_values()
    ax = success_by_type.plot(kind='barh', color='teal')
    plt.title('Success Rate by Test Type', fontsize=14)
    plt.xlabel('Success Rate')
    plt.ylabel('Test Type')
    for i, v in enumerate(success_by_type):
        ax.text(v + 0.01, i, f"{v:.0%}", va='center')
    
    # Plot 2: Test results pie chart
    plt.subplot(2, 1, 2)
    test_counts = df_validation['Success'].value_counts()
    test_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red'], 
                    labels=['Passed', 'Failed'], startangle=90)
    plt.title('Test Results', fontsize=14)
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig("data/08_reporting/data_unit_test_summary.png")
    logger.info("Saved visualization to data/08_reporting/data_unit_test_summary.png")
    
    # Plot 3: Detailed test results by column
    if 'Column' in df_validation.columns:
        plt.figure(figsize=(12, 8))
        # Use Test Description for column-wise results too
        columns_with_values = df_validation.dropna(subset=['Column'])
        if not columns_with_values.empty:
            column_results = columns_with_values.groupby(['Column', 'Test Description'])['Success'].mean().unstack()
            
            if not column_results.empty:
                column_results.plot(kind='bar', stacked=False)
                plt.title('Test Success by Column and Type', fontsize=14)
                plt.xlabel('Column')
                plt.ylabel('Success Rate')
                plt.legend(title='Test Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig("data/08_reporting/data_unit_test_by_column.png")
                logger.info("Saved column-wise results to data/08_reporting/data_unit_test_by_column.png")
    
    # Log summary statistics
    logger.info(f"Test Summary: {df_validation['Success'].sum()} passed, "
                f"{len(df_validation) - df_validation['Success'].sum()} failed "
                f"({df_validation['Success'].mean()*100:.1f}% success rate)")
    
    return df_validation