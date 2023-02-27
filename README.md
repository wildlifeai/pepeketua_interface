# Pepeketua ID GUI

This repository contains scripts and resources to develop a GUI that helps biologist identify individual Archey's
frogs (Leiopelma archeyi).

Lior Uzan carried out this work, with the support of Bar Vinograd and Victor Anton.

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

## Overview

This project runs some scripts to process previously identified frog photos, then run a GUI that helps biologists identify individual frogs. The interface uses deep learning tools behind the scenes to extract ID vectors from new frogs and compares them to previously identified and extracted ID vectors using a nearest neighbor search.

## Requirements

* Docker

## To start the app

1. Create a folder called `pepeketua_files`.
2. Place the two previous capture excel files in that directory, name them `wharerino.xls` and `pukeokahu.xls`. 
3. Place the photo zips `whareorino_a.zip` `whareorino_b.zip` `whareorino_c.zip` `whareorino_d.zip` `pukeokahu.zip` in 
   the same directory.
4. Download all model directories from 
   [here](https://drive.google.com/drive/folders/1_QeCXz151nE_tP-3MCPAq7y1NkbrGd5Q?usp=sharing) and place them in 
   the `pepeketua_files` directory as well.
5. Download [compose.yaml](https://github.com/wildlifeai/pepeketua_interface/blob/main/compose.yaml) to the directory 
   containing `pepeketua_files` (a.k.a `../pepeketua_files`)
6. Open a command line terminal at the directory containing `pepeketua_files`, and type in and execute the 
   command `docker compose up` 

## Usage

After the dockers from the previous section finish processing the old capture data, the app will be available at 
[this url][pepeketua_interface_url].  
See [this presentation](http://bit.ly/3SmUsj0) to learn about the app and how to use it.  

## How it works

- The scripts clean all frog sightings from the excel sheets, save them to a SQL server and save corresponding pictures
to a LMDB. 
- Then it extracts the id vectors from the frog images and saves those to Faiss indices, one per grid.
- Then the Streamlit server is started (GUI) and is accessible at [this url][pepeketua_interface_url]

## Extra files saved to the shared folder:

- `parse_previous_captures.log`: Log file for script outputs
- `incorrect_filepaths.csv`: All rows where there is a mismatch between the photo path and the the "Frog ID #" column.
- `missing_photos.csv`: All rows that have no "filepath" column value.

## Citation

If you use this code or its models, please cite:

Uzan L, Vinograd B, Anton V (2022). Pepeketua ID - A frog identification
app. https://github.com/wildlifeai/pepeketua_interface

## Collaborations/questions

We are working to make our work available to anyone interested. Please feel free to [contact us][contact_info] if you
have any questions.



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/wildlifeai/pepeketua_interface.svg?style=for-the-badge

[contributors-url]: https://github.com/wildlifeai/pepeketua_interface/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/wildlifeai/pepeketua_interface.svg?style=for-the-badge

[forks-url]: https://github.com/wildlifeai/pepeketua_interface/network/members

[stars-shield]: https://img.shields.io/github/stars/wildlifeai/pepeketua_interface.svg?style=for-the-badge

[stars-url]: https://github.com/wildlifeai/pepeketua_interface/stargazers

[issues-shield]: https://img.shields.io/github/issues/wildlifeai/pepeketua_interface.svg?style=for-the-badge

[issues-url]: https://github.com/wildlifeai/pepeketua_interface/issues

[license-shield]: https://img.shields.io/github/license/wildlifeai/pepeketua_interface.svg?style=for-the-badge

[license-url]: https://github.com/wildlifeai/pepeketua_interface/blob/main/LICENSE

[contact_info]: contact@wildlife.ai

[pepeketua_interface_url]: http://pepeketua-interface
