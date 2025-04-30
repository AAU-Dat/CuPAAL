# Dependencies

This project relies on the CUDD library for the implementation of ADDs, and the Storm library for parsing Prism models

There are two immediate options for developing this library. Either you

1. Install Storm on your machine, which includes CUDD (**Recommended**, Linux)
2. Use the supplied devcontainer built on the storm image (Windows)

## Usage

To use cupaal from the terminal, we support the following commands:

| Option                | Required | Description                                                            | Defaults |
|-----------------------|----------|------------------------------------------------------------------------|----------|
| `-m` / `--model`      | **Yes**  | Path to the file containing the model description                      |          |
| `-s` / `--sequences`  | **Yes**  | Path to the file containing observation sequences                      |          |
| `-i` / `--iterations` | No       | Maximum number of iterations for Baum-Welch                            | `100`    |
| `-e` / `--epsilon`    | No       | Convergence criterion; stops if likelihood change is smaller than this | `1e-2`   |
| `-t` / `--time`       | No       | Max time to run in seconds; no new iterations after this               | `240`    |
| `-o` / `--output`     | No       | Name of the file to save the resulting model                           |          |
| `-r` / `--results`    | No       | Name of the file to save the experimental results                      |          |

### Docker

To run an example experiment for comparing jajapy and cupaal, you only need the dockerfile, compose.yaml, and a provided
in this repository, and a PRISM file.

The following docker compose command will create a bind mount folder containing the results.

```shell
docker compose up --build
```
