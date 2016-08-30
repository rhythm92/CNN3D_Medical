def padImageWithMirroring( inputImage, inputImageDimensions, voxelsPerDimToPad ) :
    # inputImage shape: [batchSize, #channels#, r, c, z]
    # inputImageDimensions : [ batchSize, #channels, dim r, dim c, dim z ] of inputImage
    # voxelsPerDimToPad shape: [ num o voxels in r-dim to add, ...c-dim, ...z-dim ]
    # If voxelsPerDimToPad is odd, 1 more voxel is added to the right side.
    # r-axis
    assert np.all(voxelsPerDimToPad) >= 0
    padLeft = int(voxelsPerDimToPad[0]/2); padRight = int((voxelsPerDimToPad[0]+1)/2);
    paddedImage = T.concatenate([inputImage[:,:, int(voxelsPerDimToPad[0]/2)-1::-1 ,:,:], inputImage], axis=2) if padLeft >0 else inputImage
    paddedImage = T.concatenate([paddedImage, paddedImage[ :, :, -1:-1-int((voxelsPerDimToPad[0]+1)/2):-1, :, :]], axis=2) if padRight >0 else paddedImage
    # c-axis
    padLeft = int(voxelsPerDimToPad[1]/2); padRight = int((voxelsPerDimToPad[1]+1)/2);
    paddedImage = T.concatenate([paddedImage[:,:,:, padLeft-1::-1 ,:], paddedImage], axis=3) if padLeft >0 else paddedImage
    paddedImage = T.concatenate([paddedImage, paddedImage[:,:,:, -1:-1-padRight:-1,:]], axis=3) if padRight >0 else paddedImage
    # z-axis
    padLeft = int(voxelsPerDimToPad[2]/2); padRight = int((voxelsPerDimToPad[2]+1)/2)
    paddedImage = T.concatenate([paddedImage[:,:,:,:, padLeft-1::-1 ], paddedImage], axis=4) if padLeft >0 else paddedImage
    paddedImage = T.concatenate([paddedImage, paddedImage[:,:,:,:, -1:-1-padRight:-1]], axis=4) if padRight >0 else paddedImage

    newDimensions = [ inputImageDimensions[0],
                     inputImageDimensions[1],
                     inputImageDimensions[2] + voxelsPerDimToPad[0],
                     inputImageDimensions[3] + voxelsPerDimToPad[1],
                     inputImageDimensions[4] + voxelsPerDimToPad[2] ]
        
    return (paddedImage, newDimensions)