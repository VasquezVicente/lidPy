library(lidR)
library(terra)

#TLS data is the de heaviest, we need to read the catalog and decimate heavy
ctg1<- readLAScatalog("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/tiles")
outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/decimated/"
dir.create(outdir)
plot(ctg1)
opt_output_files(ctg1) <- paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
cleaned_ctg<- decimate_points(ctg1, random(30000))

###TLS classify the points
# classify ground pointss
outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/classified/"
dir.create(outdir)
opt_chunk_buffer(cleaned_ctg)<- 2
opt_chunk_size(cleaned_ctg)<-20
plot(cleaned_ctg,chunk=TRUE)
opt_output_files(cleaned_ctg) <- paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
classified_ctg <- classify_ground(cleaned_ctg, csf(cloth_resolution=0.3, rigidness=3L))

##create dtm
outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/ground/"
dir.create(outdir)
opt_output_files(classified_ctg)<-paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
dtm<- rasterize_terrain(classified_ctg, 0.2, tin(), pkg="terra")

#re-read the dtm
dtm<- rast("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/ground/rasterize_terrain.vrt")
plot(dtm)
#normalize the thing
classified_ctg<-readLAScatalog("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/classified/")
plot(classified_ctg, chunk=TRUE)
opt_chunk_buffer(classified_ctg)<- 2
opt_chunk_size(classified_ctg)<-20
outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/normalized/"
dir.create(outdir)
opt_output_files(classified_ctg)<-paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
normalized_ctg<- normalize_height(classified_ctg, dtm)

## create canopy height 
outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/canopy/"
dir.create(outdir)
opt_output_files(normalized_ctg)<-paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
opt_filter(normalized_ctg) <- "-drop_z_gt 50"
canopy_ctg<- rasterize_canopy(normalized_ctg,0.2, p2r(subcircle=0.20))

plot(canopy_ctg)

f_metrics <- function(Z,n) {
  
  Zcov = length(Z[Z>=2 & n==1])/length(Z[n==1])
  
  p98 = quantile(Z,0.98)
  p50 = quantile(Z,0.5)
  
  lista_metrics = list(
    H98TH = p98,
    H50TH = p50,
    COV = Zcov,
    Hmean = mean(Z),
    HSD = sd(Z),
    CV = sd(Z)/mean(Z)
  )
  return(lista_metrics)
}

#cloud metrics - plot
canopy_metrics_plot = cloud_metrics(las_sub[[1]], func = ~f_metrics(Z,ReturnNumber))

# cloud metrics - grid
canopy_metrics_grid = grid_metrics(las_norm, func = ~f_metrics(Z,ReturnNumber),30)
plot(canopy_metrics_grid)




