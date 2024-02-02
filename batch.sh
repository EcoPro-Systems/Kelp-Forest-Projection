python kelp_metrics.py -l 27 -u 37
python kelp_metrics.py -l 27 -u 32
python kelp_metrics.py -l 32 -u 37
python create_interpolated_sst_sim.py -c ssp126 -m CanESM5
python create_interpolated_sst_sim.py -c ssp585 -m CanESM5
python create_interpolated_sst_sim.py -c ssp126 -m GFDL-ESM4
python create_interpolated_sst_sim.py -c ssp585 -m GFDL-ESM4
python kelp_metrics_sim.py -c ssp126 -l 27 -u 37 -m CanESM5
python kelp_metrics_sim.py -c ssp126 -l 27 -u 32 -m CanESM5
python kelp_metrics_sim.py -c ssp126 -l 32 -u 37 -m CanESM5
python kelp_metrics_sim.py -c ssp585 -l 27 -u 37 -m CanESM5
python kelp_metrics_sim.py -c ssp585 -l 27 -u 32 -m CanESM5
python kelp_metrics_sim.py -c ssp585 -l 32 -u 37 -m CanESM5
python kelp_metrics_sim.py -c ssp126 -l 27 -u 37 -m GFDL-ESM4
python kelp_metrics_sim.py -c ssp126 -l 27 -u 32 -m GFDL-ESM4
python kelp_metrics_sim.py -c ssp126 -l 32 -u 37 -m GFDL-ESM4
python kelp_metrics_sim.py -c ssp585 -l 27 -u 37 -m GFDL-ESM4
python kelp_metrics_sim.py -c ssp585 -l 27 -u 32 -m GFDL-ESM4
python kelp_metrics_sim.py -c ssp585 -l 32 -u 37 -m GFDL-ESM4

python regressors_predict.py -f Data/kelp_metrics_27_37.pkl -fs Data/kelp_metrics_sim_27_37_ssp126_BGL.pkl
python regressors_predict.py -f Data/kelp_metrics_27_32.pkl -fs Data/kelp_metrics_sim_27_32_ssp126_BGL.pkl
python regressors_predict.py -f Data/kelp_metrics_32_37.pkl -fs Data/kelp_metrics_sim_32_37_ssp126_BGL.pkl
python regressors_predict.py -f Data/kelp_metrics_27_37.pkl -fs Data/kelp_metrics_sim_27_37_ssp585_BGL.pkl
python regressors_predict.py -f Data/kelp_metrics_27_32.pkl -fs Data/kelp_metrics_sim_27_32_ssp585_BGL.pkl
python regressors_predict.py -f Data/kelp_metrics_32_37.pkl -fs Data/kelp_metrics_sim_32_37_ssp585_BGL.pkl

python plot_timeseries.py -f Data/kelp_metrics_27_37.pkl
python plot_timeseries.py -f Data/kelp_metrics_27_32.pkl
python plot_timeseries.py -f Data/kelp_metrics_32_37.pkl
python plot_histogram_sst.py -f Data/kelp_metrics_27_37.pkl
python plot_histogram_sst.py -f Data/kelp_metrics_27_32.pkl
python plot_histogram_sst.py -f Data/kelp_metrics_32_37.pkl
python plot_histogram_kelp.py -f Data/kelp_metrics_27_37.pkl
python plot_histogram_kelp.py -f Data/kelp_metrics_27_32.pkl
python plot_histogram_kelp.py -f Data/kelp_metrics_32_37.pkl
python overplot_timeseries.py
python overplot_histogram.py
python trends_quarterly.py -f Data/kelp_metrics_27_37.pkl
python trends_quarterly.py -f Data/kelp_metrics_27_32.pkl
python trends_quarterly.py -f Data/kelp_metrics_32_37.pkl
python regressors_predict.py -f Data/kelp_metrics_27_37.pkl -fs Data/kelp_metrics_sim_27_37_ssp126_BGL.pkl
python regressors_predict.py -f Data/kelp_metrics_27_37.pkl -fs Data/kelp_metrics_sim_27_37_ssp585_BGL.pkl