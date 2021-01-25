while read p; do
	python -m torch.distributed.launch --nproc_per_node=1 edge_diag_plot.py --n_procs=1 --subsample=-1 --plot_subsample=0.1 --load_model=$p
	python -m torch.distributed.launch --nproc_per_node=1 edge_diag_plot.py --n_procs=1 --subsample=0.1 --plot_subsample=0.1 --load_model=$p
	python -m torch.distributed.launch --nproc_per_node=1 edge_diag_plot.py --n_procs=1 --subsample=0.01 --plot_subsample=0.1 --load_model=$p
done <resls
