#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int main() {
    double start,end;
    FILE *fIn = fopen("tulip.bmp", "rb");
    FILE *fOut = fopen("tulip2_gray.bmp", "wb");
    if (!fIn || !fOut) {
        printf("File error.\n");
        return 0;
    }

    unsigned char header[54];
    fread(header, sizeof(unsigned char), 54, fIn); // read the 54-byte header
    fwrite(header, sizeof(unsigned char), 54, fOut); // write the header to output file

    // extract image height and width from header
    int width = *(int*)&header[18];
    int height = abs(*(int*)&header[22]);
    int stride = (width * 3 + 3) & ~3;
    int padding = stride - width * 3;

    printf("width: %d (%d)\n", width, width * 3);
    printf("height: %d\n", height);
    printf("stride: %d\n", stride);
    printf("padding: %d\n", padding);

    unsigned char *buffer = (unsigned char*) malloc(stride);
    unsigned long long totalGray = 0; // For reduction
    int numPixels = width * height;

    start = omp_get_wtime ();

    // Parallel processing of rows
    #pragma omp parallel for reduction(+:totalGray) schedule(dynamic) 
    for (int y = 0; y < height; ++y) {
        // Read one row into buffer
        #pragma omp critical
        {
            fseek(fIn, 54 + y * stride, SEEK_SET);
            fread(buffer, stride, 1, fIn);
        }

        // Convert pixels to grayscale
        for (int x = 0; x < width; ++x) {
            int index = x * 3;
            unsigned char blue = buffer[index];
            unsigned char green = buffer[index + 1];
            unsigned char red = buffer[index + 2];
            unsigned char gray = (unsigned char)(0.11 * blue + 0.59 * green + 0.3 * red);
            buffer[index] = gray;
            buffer[index + 1] = gray;
            buffer[index + 2] = gray;
            totalGray += gray;
        }

        // Write the processed row back
        #pragma omp critical
        {
            fseek(fOut, 54 + y * stride, SEEK_SET);
            fwrite(buffer, stride, 1, fOut);
        }
    }

    free(buffer);
    fclose(fOut);
    fclose(fIn);

    end = omp_get_wtime ( );

    double averageGray = totalGray / (double)numPixels;
    printf("Average grayscale value: %f\n", averageGray);
   printf("  Elapsed wall clock time = %f",end-start);

    return 0;
}
