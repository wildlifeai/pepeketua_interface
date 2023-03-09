## To use this app, you need to upload an excel sheet with frog sightings, and the corresponding pictures.
### Excel Sheet:
The sheet must contain *at minimum* the columns:
* `Grid`- Containing the grid of the frog. Each row must have a value from the following options: `["Grid A", "Grid B", "Grid C", "Grid D", "Pukeokahu Frog Monitoring"]`
* `filepath`- This column must contain the name of, or path to, the images you are uploading. In other words, if I am uploading the image `1100-158.jpg`, then the corresponding `filepath` entry must end with that file name. For example, `Grid A/Individual Frogs/81/1100-158.jpg`, or just `1100-158.jpg`.
##### Optional Columns:
* `SVL (mm)`, `Weight (g)`, `Capture photo code` - The app can use these columns to rerank the results, and bring frogs with similar statistics to your query to the top.  
Thus, the top frog matches will not only look like your query but also share similar SVL, Weight, Capture code.
### Images
Make sure that for every image you upload, there is a corresponding row with the correct Grid and filename columns. If there is a difference, then rename the file or the `filename` column for the correct row. Make sure that row has the correct Grid name, otherwise it won't be compared to the correct images.