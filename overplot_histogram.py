from joblib import load
import matplotlib.pyplot as plt
import numpy as np

# set up figure 
fig, ax = plt.subplots(2,2, figsize=(10,10))
fig.suptitle(f"Sea Surface Temperature by Season")

temp_bins = np.linspace(10,25,31)

# sort files by latitude
files = [
    #'Data/kelp_metrics_27_37_histogram_sst.pkl', 
    'Data/kelp_metrics_27_32_histogram_sst.pkl', 
    'Data/kelp_metrics_32_37_histogram_sst.pkl']

# loop over each file and load data
for pkl in files:
    
    # load data
    data = load(pkl)
    
    # extract latitude region from name
    parts = pkl.split('_')
    region = f"{parts[2]} - {parts[3]}N"
    
    # calc metrics + add to label
    mean = data['Winter (Jan-Mar)'].mean()
    std = data['Winter (Jan-Mar)'].std()
    region += f" ({mean:.2f} +- {std:.2f})"
    
    # plot winter data
    ax[0,0].hist(data['Winter (Jan-Mar)'], bins = temp_bins, label=region,alpha=0.5)
    ax[0,0].set_title('Winter (Jan-Mar)')
    ax[0,0].set_xlabel("Temperature [C]")
    
    # TODO finish plotting for other seasons
    
    parts = pkl.split('_')
    region = f"{parts[2]} - {parts[3]}N"
    
    #spring
    mean_spring = data['Spring (Apr-Jun)'].mean()
    std_spring = data['Spring (Apr-Jun)'].std()
    region += f" ({mean_spring:.2f} +- {std_spring:.2f})"
    
    ax[0,1].hist(data['Spring (Apr-Jun)'], bins = temp_bins, label=region,alpha=0.5)
    ax[0,1].set_title('Spring (Apr-Jun)')
    ax[0,1].set_xlabel("Temperature [C]")
    
    #summer
    parts = pkl.split('_')
    region = f"{parts[2]} - {parts[3]}N"
    
    mean_summer = data['Summer (Jul-Sep)'].mean()
    std_summer = data['Summer (Jul-Sep)'].std()
    region += f" ({mean_summer:.2f} +- {std_summer:.2f})"
    
    ax[1,0].hist(data['Summer (Jul-Sep)'], bins = temp_bins, label=region,alpha=0.5)
    ax[1,0].set_title('Summer (Jul-Sep)')
    ax[1,0].set_xlabel("Temperature [C]")
    
    #fall
    parts = pkl.split('_')
    region = f"{parts[2]} - {parts[3]}N"
    
    mean_fall = data['Fall (Oct-Dec)'].mean()
    std_fall = data['Fall (Oct-Dec)'].std()
    region += f" ({mean_fall:.2f} +- {std_fall:.2f})"
    
    ax[1,1].hist(data['Fall (Oct-Dec)'], bins = temp_bins, label=region,alpha=0.5)
    ax[1,1].set_title('Fall (Oct-Dec)')
    ax[1,1].set_xlabel("Temperature [C]")

# turn on legend
ax[0,0].legend(loc='best')
ax[0,1].legend(loc='best')
ax[1,0].legend(loc='best')
ax[1,1].legend(loc='best')
plt.tight_layout()
plt.savefig('Data/overplot_histogram_sst.png')
plt.close()


# ['Winter -> Spring', 'Spring -> Summer', 'Summer -> Fall', 'Fall -> Winter']
# set up figure 
fig, ax = plt.subplots(2,2, figsize=(10,10))
fig.suptitle(f"Change in Kelp by Season")

kelp_bins = np.linspace(-1000,1000,50)

#files for kelp abundance by season
files = [
    #'Data/kelp_metrics_27_37_histogram_kelp.pkl', 
    'Data/kelp_metrics_27_32_histogram_kelp.pkl', 
    'Data/kelp_metrics_32_37_histogram_kelp.pkl']

# loop over each file and load data
for pkl in files:
    
    # load data
    data = load(pkl,'rb')
    print(data.keys())
    
    # extract latitude region from name
    parts = pkl.split('_')
    region = f"{parts[2]} - {parts[3]}N"
    
    # calc metrics + add to label
    mean = data['Winter -> Spring'].mean()
    std = data['Winter -> Spring'].std()
    region += f" ({mean:.2f} +- {std:.2f})"

    # plot winter data
    ax[0,0].hist(data['Winter -> Spring'], bins = kelp_bins, label=region,alpha=0.5)
    ax[0,0].set_title('Winter -> Spring')
    ax[0,0].set_xlabel(r"Change in Kelp [m$^2$]")

    # TODO finish plotting for other seasons

    parts = pkl.split('_')
    region = f"{parts[2]} - {parts[3]}N"

    #spring
    mean_spring = data['Spring -> Summer'].mean()
    std_spring = data['Spring -> Summer'].std()
    region += f" ({mean_spring:.2f} +- {std_spring:.2f})"

    ax[0,1].hist(data['Spring -> Summer'], bins = kelp_bins, label=region,alpha=0.5)
    ax[0,1].set_title('Spring -> Summer')
    ax[0,1].set_xlabel(r"Change in Kelp [m$^2$]")

    #summer
    parts = pkl.split('_')
    region = f"{parts[2]} - {parts[3]}N"

    mean_summer = data['Summer -> Fall'].mean()
    std_summer = data['Summer -> Fall'].std()
    region += f" ({mean_summer:.2f} +- {std_summer:.2f})"

    ax[1,0].hist(data['Summer -> Fall'], bins = kelp_bins, label=region,alpha=0.5)
    ax[1,0].set_title('Summer -> Fall')
    ax[1,0].set_xlabel(r"Change in Kelp [m$^2$]")

    #fall
    parts = pkl.split('_')
    region = f"{parts[2]} - {parts[3]}N"

    mean_fall = data['Fall -> Winter'].mean()
    std_fall = data['Fall -> Winter'].std()
    region += f" ({mean_fall:.2f} +- {std_fall:.2f})"

    ax[1,1].hist(data['Fall -> Winter'], bins = kelp_bins, label=region,alpha=0.5)
    ax[1,1].set_title('Fall -> Winter')
    ax[1,1].set_xlabel(r'Change in Kelp [m$^2$]')

# turn on legend
ax[0,0].legend(loc='best')
ax[0,1].legend(loc='best')
ax[1,0].legend(loc='best')
ax[1,1].legend(loc='best')
plt.tight_layout()
plt.savefig('Data/overplot_histogram_kelp.png')
plt.close()