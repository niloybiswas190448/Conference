# Travel Behavior Analysis - Google Colab Compatible Script
# Install required packages
!pip install semopy graphviz pandas numpy matplotlib seaborn scipy scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import semopy
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style and color palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

# Load the dataset
print("Loading Travel_Behavior_22 batch.csv...")
df = pd.read_csv('Travel_Behavior_22 batch.csv')

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Missing values per column:")
print(df.isnull().sum())

# Define columns for analysis
analysis_columns = [
    'Gender', 'Total_Family_Member', 'House_Ownership', 
    'Building_Material_Roof', 'Building_Material_Wall', 'Building_Material_Floor',
    'Family_Income', 'Vehicle_Ownership', 'Total_Trip', 'Total_Cost', 
    'Total_Distance', 'Total_Time', 'Working_Mode', 'School_Mode', 
    'Shopping_Mode', 'Recreation_Mode'
]

# Check which columns exist in the dataset
existing_columns = [col for col in analysis_columns if col in df.columns]
missing_columns = [col for col in analysis_columns if col not in df.columns]

print(f"\nColumns found in dataset: {len(existing_columns)}")
print(f"Missing columns: {missing_columns}")

# Use only existing columns for analysis
df_analysis = df[existing_columns].copy()

# DESCRIPTIVE STATISTICS
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

# Calculate descriptive statistics for numerical columns
numerical_cols = df_analysis.select_dtypes(include=[np.number]).columns
categorical_cols = df_analysis.select_dtypes(include=['object']).columns

print("\nNumerical Variables:")
print("-" * 40)
for col in numerical_cols:
    if col in existing_columns:
        data = df_analysis[col].dropna()
        print(f"\n{col}:")
        print(f"  Mean: {data.mean():.3f}")
        print(f"  Median: {data.median():.3f}")
        print(f"  Std: {data.std():.3f}")
        print(f"  Min: {data.min():.3f}")
        print(f"  Max: {data.max():.3f}")

print("\nCategorical Variables:")
print("-" * 40)
for col in categorical_cols:
    if col in existing_columns:
        print(f"\n{col}:")
        print(df_analysis[col].value_counts().head())

# VISUALIZATIONS
print("\n" + "="*60)
print("CREATING PUBLICATION-QUALITY VISUALIZATIONS")
print("="*60)

# Set up the figure with subplots
fig = plt.figure(figsize=(20, 16))

# 1. Income Distribution
if 'Family_Income' in existing_columns:
    plt.subplot(2, 3, 1)
    sns.histplot(data=df_analysis, x='Family_Income', kde=True, 
                color='steelblue', alpha=0.7, bins=30)
    plt.title('Family Income Distribution', fontweight='bold', fontsize=14)
    plt.xlabel('Family Income', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.grid(True, alpha=0.3)

# 2. Vehicle Ownership
if 'Vehicle_Ownership' in existing_columns:
    plt.subplot(2, 3, 2)
    vehicle_counts = df_analysis['Vehicle_Ownership'].value_counts()
    colors = sns.color_palette("pastel", len(vehicle_counts))
    plt.pie(vehicle_counts.values, labels=vehicle_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('Vehicle Ownership Distribution', fontweight='bold', fontsize=14)

# 3. House Ownership
if 'House_Ownership' in existing_columns:
    plt.subplot(2, 3, 3)
    sns.countplot(data=df_analysis, x='House_Ownership', 
                 palette='Set2', alpha=0.8)
    plt.title('House Ownership Distribution', fontweight='bold', fontsize=14)
    plt.xlabel('House Ownership Status', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

# 4. Transport Modes Comparison
transport_modes = ['Working_Mode', 'School_Mode', 'Shopping_Mode', 'Recreation_Mode']
existing_transport_modes = [col for col in transport_modes if col in existing_columns]

if existing_transport_modes:
    plt.subplot(2, 3, 4)
    mode_data = []
    for mode in existing_transport_modes:
        mode_counts = df_analysis[mode].value_counts()
        for transport_type, count in mode_counts.items():
            mode_data.append({'Mode_Type': mode.replace('_Mode', ''), 
                            'Transport': transport_type, 'Count': count})
    
    mode_df = pd.DataFrame(mode_data)
    sns.barplot(data=mode_df, x='Mode_Type', y='Count', hue='Transport',
               palette='Set3', alpha=0.8)
    plt.title('Transport Modes by Activity Type', fontweight='bold', fontsize=14)
    plt.xlabel('Activity Type', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.legend(title='Transport Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)

# 5. Family Size vs Total Trips
if 'Total_Family_Member' in existing_columns and 'Total_Trip' in existing_columns:
    plt.subplot(2, 3, 5)
    sns.scatterplot(data=df_analysis, x='Total_Family_Member', y='Total_Trip',
                   alpha=0.6, s=60, color='coral')
    plt.title('Family Size vs Total Trips', fontweight='bold', fontsize=14)
    plt.xlabel('Total Family Members', fontweight='bold')
    plt.ylabel('Total Trips', fontweight='bold')
    plt.grid(True, alpha=0.3)

# 6. Cost vs Distance Relationship
if 'Total_Cost' in existing_columns and 'Total_Distance' in existing_columns:
    plt.subplot(2, 3, 6)
    sns.regplot(data=df_analysis, x='Total_Distance', y='Total_Cost',
               scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.title('Travel Cost vs Distance Relationship', fontweight='bold', fontsize=14)
    plt.xlabel('Total Distance', fontweight='bold')
    plt.ylabel('Total Cost', fontweight='bold')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# STRUCTURAL EQUATION MODELING
print("\n" + "="*60)
print("STRUCTURAL EQUATION MODELING (SEM)")
print("="*60)

# Prepare data for SEM - convert categorical variables to numerical if needed
sem_data = df_analysis.copy()

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in categorical_cols:
    if col in sem_data.columns:
        sem_data[col] = le.fit_transform(sem_data[col].astype(str))

# Remove rows with missing values for SEM
sem_data_clean = sem_data.dropna()

print(f"SEM dataset shape after cleaning: {sem_data_clean.shape}")

# Define the SEM model specification
model_spec = '''
# Measurement model (latent constructs)
Sociodemographic =~ Gender + Total_Family_Member
Housing =~ House_Ownership + Building_Material_Roof + Building_Material_Wall + Building_Material_Floor
Economic =~ Family_Income + Vehicle_Ownership
Travel_Activity =~ Total_Trip + Total_Cost + Total_Distance + Total_Time
Accessibility =~ Working_Mode + School_Mode + Shopping_Mode + Recreation_Mode

# Structural model
Travel_Behavior =~ Sociodemographic + Housing + Economic + Travel_Activity + Accessibility
'''

# Filter model specification based on available columns
available_indicators = []
model_lines = model_spec.strip().split('\n')
filtered_model_lines = []

for line in model_lines:
    if '=~' in line and not line.strip().startswith('#'):
        parts = line.split('=~')
        if len(parts) == 2:
            latent = parts[0].strip()
            indicators = [ind.strip() for ind in parts[1].split('+')]
            available_inds = [ind for ind in indicators if ind in existing_columns]
            
            if len(available_inds) >= 2:  # Need at least 2 indicators per latent variable
                filtered_model_lines.append(f"{latent} =~ {' + '.join(available_inds)}")
                available_indicators.extend(available_inds)
    elif line.strip().startswith('#') or line.strip() == '':
        filtered_model_lines.append(line)

# Create final model specification
final_model_spec = '\n'.join(filtered_model_lines)

print("Final SEM Model Specification:")
print(final_model_spec)

# Prepare data with only available indicators
sem_final_data = sem_data_clean[available_indicators].copy()

# Standardize the data for better convergence
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sem_final_data_scaled = pd.DataFrame(
    scaler.fit_transform(sem_final_data),
    columns=sem_final_data.columns
)

try:
    # Fit the SEM model
    print("\nFitting SEM model...")
    model = semopy.Model(final_model_spec)
    results = model.fit(sem_final_data_scaled)
    
    print("\n" + "="*50)
    print("MODEL FIT INDICES")
    print("="*50)
    
    # Calculate and display fit indices
    fit_indices = semopy.calc_stats(model)
    
    print(f"Chi-square: {fit_indices.loc['chi2', 'Value']:.3f}")
    print(f"Degrees of freedom: {fit_indices.loc['dof', 'Value']:.0f}")
    print(f"P-value: {fit_indices.loc['pvalue', 'Value']:.3f}")
    print(f"CFI: {fit_indices.loc['CFI', 'Value']:.3f}")
    print(f"TLI: {fit_indices.loc['TLI', 'Value']:.3f}")
    print(f"RMSEA: {fit_indices.loc['RMSEA', 'Value']:.3f}")
    print(f"SRMR: {fit_indices.loc['SRMR', 'Value']:.3f}")
    print(f"AIC: {fit_indices.loc['AIC', 'Value']:.3f}")
    print(f"BIC: {fit_indices.loc['BIC', 'Value']:.3f}")
    
    print("\n" + "="*50)
    print("STANDARDIZED PATH COEFFICIENTS")
    print("="*50)
    
    # Get parameter estimates
    estimates = model.inspect()
    
    # Display standardized estimates
    standardized_estimates = estimates[estimates['op'] == '=~'].copy()
    standardized_estimates = standardized_estimates[['lval', 'rval', 'Estimate', 'Std. Err', 'z-value', 'p-value']]
    
    print("\nFactor Loadings:")
    print(standardized_estimates.to_string(index=False))
    
    # Structural relationships
    structural_estimates = estimates[estimates['op'] == '~'].copy()
    if not structural_estimates.empty:
        print("\nStructural Relationships:")
        structural_estimates = structural_estimates[['lval', 'rval', 'Estimate', 'Std. Err', 'z-value', 'p-value']]
        print(structural_estimates.to_string(index=False))
    
    print("\n" + "="*50)
    print("PATH DIAGRAM")
    print("="*50)
    
    # Create path diagram
    try:
        # Use semopy's built-in plotting
        g = semopy.semplot(model, "path_diagram.png", plot_covs=True)
        print("Path diagram saved as 'path_diagram.png'")
        
        # Alternative: Create a simple visualization using matplotlib
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.9, 'SEM Path Diagram', ha='center', va='center', 
                fontsize=20, fontweight='bold', transform=plt.gca().transAxes)
        
        # Display key relationships as text
        y_pos = 0.8
        plt.text(0.5, y_pos, 'Latent Constructs and Indicators:', ha='center', va='center',
                fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        
        y_pos -= 0.1
        for _, row in standardized_estimates.iterrows():
            text = f"{row['lval']} → {row['rval']}: {row['Estimate']:.3f}"
            plt.text(0.5, y_pos, text, ha='center', va='center',
                    fontsize=11, transform=plt.gca().transAxes)
            y_pos -= 0.05
            
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not create graphical path diagram: {e}")
        print("Model structure summary provided above.")

except Exception as e:
    print(f"SEM model fitting failed: {e}")
    print("This might be due to:")
    print("1. Insufficient data or too many missing values")
    print("2. Model identification issues")
    print("3. Convergence problems")
    print("\nTrying a simplified model...")
    
    # Try a simplified model with available numerical variables
    numerical_available = [col for col in numerical_cols if col in existing_columns]
    
    if len(numerical_available) >= 4:
        simplified_model = f'''
        Factor1 =~ {" + ".join(numerical_available[:len(numerical_available)//2])}
        Factor2 =~ {" + ".join(numerical_available[len(numerical_available)//2:])}
        '''
        
        try:
            simple_model = semopy.Model(simplified_model)
            simple_results = simple_model.fit(sem_final_data_scaled[numerical_available])
            
            print("\nSimplified Model Fit Indices:")
            simple_fit = semopy.calc_stats(simple_model)
            print(f"Chi-square: {simple_fit.loc['chi2', 'Value']:.3f}")
            print(f"CFI: {simple_fit.loc['CFI', 'Value']:.3f}")
            print(f"RMSEA: {simple_fit.loc['RMSEA', 'Value']:.3f}")
            
        except Exception as e2:
            print(f"Simplified model also failed: {e2}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("Summary:")
print(f"- Dataset loaded with {df.shape[0]} observations and {df.shape[1]} variables")
print(f"- Descriptive statistics calculated for {len(existing_columns)} variables")
print(f"- Publication-quality visualizations created")
print(f"- SEM analysis attempted with available variables")