import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import matplotlib.pyplot as plt
import seaborn as sns
import os

def access_data_unit_test_report():
    """
    Access and display the data unit test report from the Kedro memory dataset.
    """
    # Initialize the Kedro session
    project_path = os.path.dirname(os.path.abspath(__file__))
    metadata = bootstrap_project(project_path)
    session = KedroSession.create(metadata.package_name, project_path)
    context = session.load_context()

    try:
        # Try to access the memory dataset
        report = context.catalog.load("reporting_data_unit_test")
        print("Successfully loaded the data unit test report!")
        
        # Create output directory if it doesn't exist
        os.makedirs("data/08_reporting", exist_ok=True)
        
        # Save the report to CSV for later access
        report.to_csv("data/08_reporting/data_unit_test_results.csv", index=False)
        print(f"Report saved to data/08_reporting/data_unit_test_results.csv")
        
        # Generate basic visualization
        plt.figure(figsize=(12, 6))
        
        # Plot success rate by expectation type
        plt.subplot(1, 2, 1)
        success_by_type = report.groupby('Expectation Type')['Success'].mean()
        success_by_type.plot(kind='bar', color='teal')
        plt.title('Success Rate by Expectation Type')
        plt.xlabel('Expectation Type')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45, ha='right')
        
        # Plot count of tests by result
        plt.subplot(1, 2, 2)
        test_counts = report['Success'].value_counts()
        test_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red'])
        plt.title('Test Results')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.savefig("data/08_reporting/test_results_summary.png")
        print(f"Visualization saved to data/08_reporting/test_results_summary.png")
        
        # Return the report for further inspection
        return report
        
    except Exception as e:
        print(f"Error accessing data unit test report: {e}")
        print("Ensure you've run the data_unit_tests pipeline before running this script.")
        return None

if __name__ == "__main__":
    report = access_data_unit_test_report()
    if report is not None:
        print("\nReport Summary:")
        print(f"Total tests: {len(report)}")
        print(f"Passed tests: {report['Success'].sum()}")
        print(f"Failed tests: {len(report) - report['Success'].sum()}")
        print(f"Success rate: {report['Success'].mean()*100:.2f}%")
        
        # Display the first few rows of the report
        print("\nReport Preview:")
        print(report.head())