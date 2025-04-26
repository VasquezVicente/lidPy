#remotes::install_github('bi0m3trics/spanner')
#remotes::install_github('lucasbielak/lidUrb')
library(lidR)
library(spanner)
library(lidUrb)
library(reticulate)
virtualenv_create("lidUrb", packages="jakteristics", pip= TRUE)
use_virtualenv("lidUrb", required=TRUE)
#recall 1.chunkdatav2.py which chunks into 25 tiles with 5 meter buffer in all sides
set_lidr_threads(8)
tiles<-list.files("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1",pattern = "\\.laz$", full.names = TRUE)

# read the first tile corresponding to Q20 3020
Q3020<-readLAS(tiles[1])
Q3020<- classify_noise(Q3020, sor(k=30, m=5, quantile=FALSE))
Q3020<- classify_ground(Q3020,csf(class_threshold=0.1,cloth_resolution=0.3, rigidness=2L))
Q3020<- normalize_height(Q3020, tin())
rm(Q3020)
# I belive cloud was already normalized. No reason to normalize once more

#plot(filter_poi(Q3020, Classification==2 & Classification!=LASNOISE)) #plot the ground points

# we will remove the ground points and noisy points 
Q3020_filtered<- filter_poi(Q3020, Classification!=LASNOISE & Classification!=2)
#plot(Q3020_filtered)
#Q3020_segmented<- LW_segmentation_dbscan(Q3020_filtered)
Q3020_segmented_graph<- LW_segmentation_graph(Q3020_filtered)

plot(Q3020_filtered, color="Z", legend=TRUE)
cleaned <- filter_poi(Q3020_segmented, p_wood ==1) 
plot(cleaned, color="Z", legend=TRUE)
 
#preprocessing before segmentation is neccesary


# classify the ground points

myTreeLocs = get_raster_eigen_treelocs(las = cleaned, res = 0.01,
                                        pt_spacing = 0.01,
                                        dens_threshold = 0.1,
                                        neigh_sizes = c(0.333, 0.166, 0.5),
                                        eigen_threshold = 0.6,
                                        grid_slice_min = 1,
                                        grid_slice_max = 2,
                                        minimum_polygon_area = 0.01,
                                        cylinder_fit_type = "ransac",
                                        max_dia = 1,
                                        SDvert = 0.25,
                                        n_pts = 20,
                                        n_best = 25,
                                        inliers= 0.9,
                                        conf= 0.99,
                                        max_angle=20)

plot(grid_canopy(cleaned, res= 0.2, p2r()))
symbols(st_coordinates(myTreeLocs)[,1],st_coordinates(myTreeLocs)[,2],
circles = myTreeLocs$Radius^2, inches=FALSE, add=TRUE, bg='black')

myTreeGraph = segment_graph(las = cleaned, tree.locations = myTreeLocs, k = 50,
                              distance.threshold = 0.5,
                              use.metabolic.scale = FALSE,
                              ptcloud_slice_min = 0.6666,
                              ptcloud_slice_max = 2.0,
                              subsample.graph = 0.1,
                              return.dense = TRUE)

plot(myTreeGraph, color="treeID", pal=spanner_pal())

myTreeLocs <- myTreeLocs[order(-myTreeLocs$Radius), ]
head(myTreeLocs)

filtered_las<- filter_poi(myTreeGraph, treeID==26)
plot(filtered_las)

