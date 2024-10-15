## Willow Flycatcher Distribution

<img src="willow-flycatcher.png" alt="Willow Flycatcher" width="720" height="550" longdesc="https://macaulaylibrary.org/asset/451259001" /> 

#### Species Description
Willow Flycatchers (*Empidonax traillii*) specialize in areas with willows and other shrubs [near running and still water](https://www.audubon.org/field-guide/bird/willow-flycatcher). They are about 6 inches in length with brown, gray, and white plumage, a rounded wing, and a square-tipped tail. Calls are in the form of a chirp, buzz, or trill and match an undulating pattern. Nests are placed 4-15 feet above water or damp ground and constructed as an open cup of grass, bark, and plant fibers. The species migrates long distances, breeding in the U.S. and Canada and wintering in Mexico, Central America, and northern South America. They are common in most locations in their range despite a [25% decline](https://www.allaboutbirds.org/guide/Willow_Flycatcher/lifehistory) in population between 1966 to 2019. The loss of wet marshes, wet meadows, and riparian vegetation has contributed to [declining species abundance](https://www.fs.usda.gov/detail/tahoe/landmanagement/resourcemanagement/?cid=stelprdb5357314#:~:text=The%20scientific%20name%20for%20willow,t). According to the Bird Genoscape Project, there are seven geneticially [distinct populations](https://www.birdgenoscape.org/willow-flycatcher/) of Willow Flycatcher in North America: the Pacific Northwest, Kern, California, southern California, White Mountain, Arizona, Interior West, Southwest, and Eastern. However, there are only four recognized subspecies: *E. t. brewsteri*, *E. t. adastus*, *E. t. extimus*, and *E. t. traillii*. 

In my home state of California, there are three [endangered subspecies](https://www.fs.usda.gov/detail/tahoe/landmanagement/resourcemanagement/?cid=stelprdb5357314#:~:text=The%20scientific%20name%20for%20willow,t): Southwestern Willow Flycatcher in central and southern California (Federal and State), Little Willow Flycatcher in high elevation Sierra Nevada (State), and Great Basin Willow Flycatcher in desert riparian area (State). Researchers have found that the Southwestern Willow Flycatcher has a higher prevalence of gene variants today compared to 100 years ago that are associated with [adapting to wet and humid conditions](https://www.allaboutbirds.org/news/endangered-willow-flycatchers-in-san-diego-are-adapting-to-climate-change/). This difference is likely due to interbreeding with species in the Southwest and Pacific Northwest, producing an evolutionary response to climate change. Adaptations like these are why it is vital to preserve the interconnectivity of species populations through the protection of habitat and landscape mobility.

#### Data Description

The [GBIF Occurrence dataset](https://doi.org/10.15468/dl.jqrwjf) was retrived from the Global Biodiversity Information Facility Occurrence Store and is scoped to the year of interest (2023) and species under observation. There are 110,725 species occurrences total which are indicated by geographic coordinates spanning across five aggregated datasets. 

The [RESOLVE Ecoregions dataset](https://developers.google.com/earth-engine/datasets/catalog/RESOLVE_ECOREGIONS_2017) (2017) depicts Earth's 846 terrestrial ecoregions and was obtained as a shapefile. Ecoregions represent boundaries formed by biotic and abiotic conditions: geology, landforms, soils, vegetation, land use, wildlife, climate, and hydrology.

#### Data Citations

Global Biodiversity Information Facility. (2024). *GBIF Occurrence Download* [Data set]. https://doi.org/10.15468/dl.jqrwjf

RESOLVE. (2017). *RESOLVE Ecoregions dataset* [Data set]. https://doi.org/10.1093/biosci/bix014 

#### Methods

The occurrences data was accessed with the Python client for the [GBIF API](https://techdocs.gbif.org/en/openapi/v1/occurrence#/) and queried for the year (2023), species, and coordinates. The occurrences CSV file was ingested using the [pandas](https://pandas.pydata.org/) library and country, state/province, latitude, longitude, month, and year records were selected. The resulting DataFrame was converted to a GeoDataFrame with monthly scope using [geopandas](https://geopandas.org/en/stable/), providing coordinates as the geometry and the WGS84 projection, which represents the Earth as a 3D ellipsoid, as the coordinate reference system (CRS).

Data for ecoregions were gathered from a [RESOLVE](https://www.resolve.ngo/projects/ecoregions-world) shapefile (2017) and read into a GeoDataFrame with geopandas. This data was joined spatially with occurrences on the month and name, matching the WGS84 CRS. Monthly regional observations were counted from this GeoDataFrame. Next, the mean by region and by month was calculated and used for normalization. The data was normalized by space (ecoregion average) and time (monthly average) to account for the sampling effort.

For visualization, the GeoDataFrame was simplified to a Mercator projection from the [Cartopy](https://scitools.org.uk/cartopy/docs/latest/) library which is compatible with the [hvplot API](https://hvplot.holoviz.org/) and [GeoViews](https://geoviews.org/). The GeoDataFrame was also joined with the normalized occurrences data. The plot produced highlights monthly migration patterns and is interactive due to the sliding widget from the HoloViews [panel](https://panel.holoviz.org/reference/panes/HoloViews.html) library.


<embed type="text/html" src="willow-flycatcher-migration.html" width="600" height="800">

#### Species Distribution

The *Willow Flycatcher Migration* plot demonstrates changes in Willow Flycatcher distribution as the result of migration patterns across a single year (2023). Timing in migration reflects the annual cycles of breeding and wintering. Typically this species will journey between [2,000 to over 5,000 miles per year](https://doi.org/10.2737/RMRS-GTR-60) in keeping with these cycles. [Dates of migration]((https://doi.org/10.2737/RMRS-GTR-60)) (arrival and departure) vary depending on the latitude. In the early 1990s, spring arrivals were indicated to be near 30-35° North around late April and early May. At 46-50° North, the spring arrival window was between late May and mid-June. Fall departure dates were around late August and early October at 30-35° North whereas in the 46-50° North range the dates were between late August and late September. 

The northern limit of the species distribution and as a consequence, migration dates, are likely to move in response to climate change. Between the late 1960s and early 2000s, the northern hemisphere spring maximum temperatures increased about 1°C and during that period a [mean shift north in breeding range](https://doi.org/10.1111/j.1523-1739.2006.00609.x) for the Willow Flycatcher was observed (135.44 ± 59.37 km). Moreover, global temperatures are continuing to increase; 2023 was approximately [1.36°C warmer](https://climate.nasa.gov/vital-signs/global-temperature) than the preindustrial average (1850-1900). The 2023 migration plot illustrates a latitude realignment compared to the early 2000s in the distribution corresponding to temperature signals. Spring arrivals in April approached 45° North and fall departures in the 46-50° North range were closer to October. Within the breeding range, there were no observations for April 2023 in Canada. The earliest spring arrivals appeared in May with the greatest in Ontario about 6 degrees north of the 1990s latitude. Between 46 and 50 degrees north latitude, few observations were recorded in May at 46° North in the Eastern Canadian Forest-Boreal transition compared to the largest occurrences near the top of the range at 49° North within the British Columbia coastal conifer forests. April arrivals in the United States were most prominent near 37° North in the Interior Plateau US Hardwood Forests ecoregion. During May the lowest observations were recorded in the 30-35° North range (Arizona Mountains forests) and the highest occurrences surpassed that range at 39° North (Appalachian-Blue Ridge forests). The flycatcher distribution is strongly correlated with temperature and as evidenced by these observations, is likely expanding its northern limit and migration timing in response to climate change. 

#### References

Afzal, P. (2024, January 4). *Endangered Willow Flycatchers in San Diego are adapting to climate change.* Cornell Lab of Ornithology. https://www.allaboutbirds.org/news/endangered-willow-flycatchers-in-san-diego-are-adapting-to-climate-change 

Bird Genoscape Project. (n.d.). *Willow Flycatcher.* https://www.birdgenoscape.org/willow-flycatcher 

Chamberlain, S. (2024). *pygbif* (Version 0.6.4) [Computer software]. GitHub. https://github.com/gbif/pygbif/releases/tag/v0.6.4

Cornell Lab of Ornithology. (n.d.). *Willow Flycatcher life history.* All About Birds. https://www.allaboutbirds.org/guide/Willow_Flycatcher/lifehistory

Dinerstein, E., Olson, D., Joshi, A., Vynne, C., Burgess, N. D., Wikramanayake, E., … Saleem, M. (2017). An ecoregion-based approach to protecting half the terrestrial realm. *BioScience, 67*(6), 534–545. https://doi.org/10.1093/biosci/bix014

Finch, D. M. & Stoleson, S. H. (2000). Status, ecology, and conservation of the southwestern willow flycatcher. U.S. Department of Agriculture, Forest Service, Rocky Mountain Research Station. https://doi.org/10.2737/RMRS-GTR-60

Hitch, A. & Leberg, P. (2007). Breeding distributions of North American bird species moving north as a result of climate change. *Conservation Biology, 21*(2), 534-539. https://doi.org/10.1111/j.1523-1739.2006.00609.x

Jordahl, K., Van den Bossche, J., Fleischmann, M., Wasserman, J., McBride, J., Gerard, J., … Leblanc, F. (2024). *geopandas/geopandas: v1.0.1* (Version 1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.12625316 

Met Office. (2024). *Cartopy: a cartographic python library with a Matplotlib interface* (Version 0.24.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.13905945

NASA. (n.d.). *Global temperature*. https://climate.nasa.gov/vital-signs/global-temperature

National Audubon Society. (n.d.). *Willow Flycatcher.* Audubon. https://www.audubon.org/field-guide/bird/willow-flycatcher

Python Software Foundation. (2024). *Python* (Version 3.12.6) [Computer software]. https://docs.python.org/release/3.12.6 

Rudiger, P., Hansen, S. H., Bednar, J. A., Steven, J., Liquet, M., Little, B., … Bampton, J. (2024). *holoviz/geoviews: Version 1.13.0* (Version 1.13.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.13782761

Rudiger, P., Liquet, M., Signell, J., Hansen, S. H., Bednar, J. A., Madsen, M. S., … Hilton, T. W. (2024). *holoviz/hvplot: Version 0.11.0* (Version 0.11.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.13851295 

Rudiger, P., Hansen, S. H.,  Madsen, M. S., & Wu, J. (2024). *holoviz/panel: Version 1.5.2* (Version 1.5.2) [Computer software]. GitHub. https://github.com/holoviz/panel/releases/tag/v1.5.2

Tahoe National Forest. (n.d.). *Willow Flycatcher - introduction.* U.S. Forest Service. https://www.fs.usda.gov/detail/tahoe/landmanagement/resourcemanagement/?cid=stelprdb5357314#:~:text=The%20scientific%20name%20for%20willow,t 

The pandas development team. (2024). *pandas-dev/pandas: Pandas* (Version 2.2.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.3509134
