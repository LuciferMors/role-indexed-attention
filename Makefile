.PHONY: all clean test battery interpret plots paper

PY := python3

all: battery interpret plots paper

battery:
	$(PY) -m experiments.run_battery --seeds 0 1 2 --n_train 150000 \
	    --experiments main param_matched sample_eff

interpret:
	$(PY) -m experiments.interpret

plots:
	$(PY) -m experiments.make_plots

paper:
	$(PY) -m experiments.fill_paper

test:
	$(PY) -c "from ava import AvacchedakaAttention, AvaConfig; \
	          import torch; \
	          a = AvacchedakaAttention(AvaConfig(d_model=64, n_aspects=4, n_relations=4)); \
	          x = torch.randn(2, 8, 64); \
	          y, e = a(x, return_edges=True); \
	          assert y.shape == (2, 8, 64); \
	          assert e.shape == (2, 4, 8, 4, 8, 4); \
	          print('OK')"

clean:
	rm -f results/*.jsonl results/*.json
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.pdf paper/*.bbl paper/*.blg
