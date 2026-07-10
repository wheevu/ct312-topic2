  #!/usr/bin/env bash
  set -e

  docker compose up -d nginx

  streamlit run app/streamlit_app.py \
    --server.port 8080 \
    --server.address 0.0.0.0 \
    --server.headless true
