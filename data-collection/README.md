# Data collection

This is a simple node project, that calls the https://deathtimeline.com api and parses deaths, killers and other metadata for needs
of csv dataset file construction for the link GOT link prediction project.

## How to use

When in this folder, first run `npm install` to install all necessary node modules. Then, run the script by:

```bash
node data_collector.js > ../data/deaths.csv
```

This will create a csv file in the folder `data` in the root directory (named `deaths.csv`) containing all deaths along with
the associated metadata returned by the https://deathtimeline.com API.

## Requirements

- Node.js version 8 or bigger. (Though the script was developed for Node.js v11.0.0, it should run on v8+)
