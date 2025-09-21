run:
\tstreamlit run demo/app.py

bench:
\tpython examples/bench.py

profile-nsys:
\tnsys profile -o mart_demo --trace=cuda,osrt python examples/bench.py

profile-ncu:
\tncu --set full --target-processes all python examples/bench.py

docker-build:
\tdocker build -t rbyrct-demo .

docker-run:
\tdocker run --gpus all -p 8501:8501 rbyrct-demo

