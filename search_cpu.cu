#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define DIM 3
#define GRID 16
#define VALIDATE 10

// function declarations
void validate_grid (const float *c, const float *intervals, const int *grid_c,
                    const int *points_block_c, int D);

void validate_search (const float *q, const float *c, const int *closest, int N,
                      int D);
void write_file (double time_var, const char *filename, const char *mode);

/**
 * Find grid location of each block with gpu
 * @method find_grid_loc_gpu
 * @param  points            points matrix
 * @param  grid_loc          grid location for each point result
 * @param  n                 num of elements
 * @param  d                 grid dimension (cube)
 */
void
find_grid_loc (float *points, int *grid_loc, int n, int d)
{
  int x, y, z;
  int dd = d * d;
  for (int i = 0; i < n; i++)
    {
      x = (int) (points[i * DIM + 0] * d);
      y = (int) (points[i * DIM + 1] * d);
      z = (int) (points[i * DIM + 2] * d);
      grid_loc[i] = x + d * y + dd * z;
    }
}

void
init_rand_points (float *p, int n)
{
  int i;
  for (i = 0; i < n * DIM; i++)
    p[i] = (rand () % 1000000 / (float) (1000001));
}

/**
 * [search_block description]
 * @method search_block
 * @param  closest          current closest point index
 * @param  current_min      current closest point distance
 * @param  block_offset     block location in point array
 * @param  q                point
 * @param  block            block point array
 * @param  points_in_block  num of points the current block
 */
void
search_block (int *closest, float *current_min, int block_offset, float *q,
              float *block, int points_in_block)
{

  float dist;
  for (int i = 0; i < points_in_block; i++)
    {

      dist = (block[i * DIM + 0] - q[0]) * (block[i * DIM + 0] - q[0]);
      dist += (block[i * DIM + 1] - q[1]) * (block[i * DIM + 1] - q[1]);
      dist += (block[i * DIM + 2] - q[2]) * (block[i * DIM + 2] - q[2]);
      dist = sqrtf (dist);
      if (dist < *current_min)
        {
          *current_min = dist;
          *closest = i + block_offset;
        }
    }
}
/**
 * find closet point in c of each point in q with cpu
 * @method voidsearch
 * @param  q                [q points]
 * @param  c                [c points]
 * @param  grid             [c points location of each grid block]
 * @param  points_per_block [points in each grid block]
 * @param  closests         [result array index]
 * @param  mindists         [result array min dist found]
 * @param  N                [number of elements]
 * @param  d                [grid dimension cube]
 */
void
search (float *q, float *c, int *grid, int *points_per_block, int *closests,
        float *mindists, int N, int d)
{

  int x, y, z;
  int grid_loc;
  float b;
  int stage = 0, finished = 0;
  float block_size = 1 / (float) d;
  float point[3];

  for (int i = 0; i < N; i++)
    {
      /*if(i % 100 == 0){
    float per = (float) i / (float) N * 100;
       printf("%f\n",per  );
     } */
      point[0] = q[i * DIM + 0];
      point[1] = q[i * DIM + 1];
      point[2] = q[i * DIM + 2];

      x = (int) (point[0] * (float) d);
      y = (int) (point[1] * (float) d);
      z = (int) (point[2] * (float) d);

      grid_loc = x + d * y + d * d * z;

      mindists[i] = (1 << 10); // Inf

      search_block (&closests[i], &mindists[i], grid[grid_loc], point,
                    &c[grid[grid_loc] * DIM], points_per_block[grid_loc]);

      finished = 0;
      stage = 0;

      while (!finished)
        {
          // if(stage > 2 ) printf("%d\n",stage );
          finished = 1;

          //-------X---------
          if (x + stage + 1 < d)
            {
              b = block_size * (x + stage + 1) - point[0];
              if (b < mindists[i])
                finished = 0;
            }
          if (x - stage - 1 > -1)
            {
              b = point[0] - block_size * (x - stage);
              if (b < mindists[i])
                finished = 0;
            }
          //-------Y---------
          if (y + stage + 1 < d)
            {
              b = block_size * (y + stage + 1) - point[1];
              if (b < mindists[i])
                finished = 0;
            }
          if (y - stage - 1 > -1)
            {
              b = point[1] - block_size * (y - stage);
              if (b < mindists[i])
                finished = 0;
            }
          //-------Z---------
          if (z + stage + 1 < d)
            {
              b = block_size * (z + stage + 1) - point[2];
              if (b < mindists[i])
                finished = 0;
            }
          if (z - stage - 1 > -1)
            {
              b = point[2] - block_size * (z - stage);
              if (b < mindists[i])
                finished = 0;
            }

          stage++;

          //  if (stage == 1)
          //    finished = 0;

          if (!finished)
            {

              for (int dx = x - stage; dx <= x + stage; dx++)
                {
                  for (int dy = y - stage; dy <= y + stage; dy++)
                    {
                      for (int dz = z - stage; dz <= z + stage; dz++)
                        {

                          if (dx == x - stage || dx == x + stage
                              || dy == y - stage || dy == y + stage
                              || dz == z - stage || dz == z + stage)
                            {
                              grid_loc = dx + d * dy + d * d * dz;
                              if (dx > -1 && dx < d && dy > -1 && dy < d
                                  && dz > -1 && dz < d)
                                search_block (&closests[i], &mindists[i],
                                              grid[grid_loc], point,
                                              &c[grid[grid_loc] * DIM],
                                              points_per_block[grid_loc]);
                            }
                        }
                    }
                }
            }
        }
    }
}

float *
rearrange (float *p, int *intex, int *points_per_block, int *grid, int n, int k)
{

  for (int i = 0; i < k; i++)
    points_per_block[i] = 0;

  for (int i = 0; i < n; i++)
    {
      points_per_block[intex[i]]++;
    }

  grid[0] = 0;
  grid[1] = points_per_block[0];

  for (int i = 2; i < k; i++)
    {
      grid[i] = grid[i - 1] + points_per_block[i - 1];
    }

  int *positions = (int *) malloc (k * sizeof (int));

  for (int i = 0; i < k; i++)
    {
      positions[i] = grid[i];
    }

  float *arrangedpoints = (float *) malloc (n * DIM * sizeof (float));

  int pos;
  int posDim = 0, iDim = 0;
  for (int i = 0; i < n; i++)
    {
      pos = positions[intex[i]];
      posDim = pos * DIM;
      arrangedpoints[posDim + 0] = p[iDim + 0];
      arrangedpoints[posDim + 1] = p[iDim + 1];
      arrangedpoints[posDim + 2] = p[iDim + 2];
      iDim = iDim + DIM;
      positions[intex[i]]++;
    }
  free (p);
  free (positions);
  return arrangedpoints;
}

int
main (int argc, char **argv)
{
  if (argc != 3)
    {
      printf ("Invalid argument\n");
      exit (1);
    }
  int NQ = 1 << atoi (argv[1]);
  int NC = 1 << atoi (argv[1]);
  int N = NQ;
  int D = 1 << atoi (argv[2]);
  /*
    write_file (atoi (argv[1]), "problem_size.data", "a");
    write_file (atoi (argv[2]), "grid_size.data", "a");
  */
  int block_num = D * D * D;
  printf ("NQ=%d NC=%d D=%d block_num=%d\n", NQ, NC, D, block_num);

  float *intervals = (float *) malloc (D * sizeof (float));
  for (int i = 1; i <= D; i++)
    intervals[i - 1] = 1 / (float) D * i;

  struct timeval startwtime, endwtime;
  double elapsed_time;

  float *q, *c;
  int *grid_q, *grid_c;
  int *q_block, *c_block;
  int *points_block_q, *points_block_c;
  int *closest;
  float *mindists;
  // malloc points
  q = (float *) malloc (N * DIM * sizeof (float));
  c = (float *) malloc (N * DIM * sizeof (float));
  // malloc location of grid block in array q/c
  grid_q = (int *) malloc (block_num * sizeof (int));
  grid_c = (int *) malloc (block_num * sizeof (int));
  // malloc grid of each point
  q_block = (int *) malloc (N * sizeof (int));
  c_block = (int *) malloc (N * sizeof (int));
  // malloc points per block
  points_block_q = (int *) malloc (block_num * sizeof (int));
  points_block_c = (int *) malloc (block_num * sizeof (int));

  closest = (int *) malloc (N * sizeof (int));
  mindists = (float *) malloc (N * sizeof (float));

  init_rand_points (q, N);
  init_rand_points (c, N);

  // find_grid_loc(q,q_block,N,D);

  //-------------REARRANGE POINTS IN GRID IN CPU-----------------------
  gettimeofday (&startwtime, NULL);

  find_grid_loc (c, c_block, N, D);
  c = rearrange (c, c_block, points_block_c, grid_c, N, block_num);

  gettimeofday (&endwtime, NULL);

  elapsed_time = (double) ((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6
                           + endwtime.tv_sec - startwtime.tv_sec);

  printf ("Rearrange time : %f\n", elapsed_time);
  // write_file (elapsed_time, "rearrange_time.data", "a");

  //---------------GRID VALIDATION IN  // CPU-----------------------
  validate_grid (c, intervals, grid_c, points_block_c, D);

  //---------------SEARCH GRID IN CPU-----------------------

  gettimeofday (&startwtime, NULL);

  search (q, c, grid_c, points_block_c, closest, mindists, N, D);

  gettimeofday (&endwtime, NULL);

  elapsed_time = (double) ((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6
                           + endwtime.tv_sec - startwtime.tv_sec);
  printf ("Search Time CPU : %f\n", elapsed_time);
  //  write_file (elapsed_time, "search_cpu_time.data", "a");
  validate_search (q, c, closest, N, D);

  //---------------CLEAN UP-------------------------------------

  free (q);
  free (c);
  free (grid_c);
  free (grid_q);
  free (c_block);
  free (q_block);
  free (points_block_c);
  free (points_block_q);
  free (closest);
  free (mindists);
}
void
validate_grid (const float *c, const float *intervals, const int *grid_c,
               const int *points_block_c, int D)
{

  int sum = 0;
  int fails = 0;
  float xmax, ymax, zmax, xmin, ymin, zmin;
  int pos, block_pos, point_pos;

  for (int x = 0; x < D; x++)
    {
      xmax = intervals[x];
      if (x == 0)
        {
          xmin = 0;
        }
      else
        {
          xmin = intervals[x - 1];
        }
      for (int y = 0; y < D; y++)
        {
          ymax = intervals[y];
          if (x == 0)
            {
              ymin = 0;
            }
          else
            {
              ymin = intervals[y - 1];
            }
          for (int z = 0; z < D; z++)
            {
              zmax = intervals[z];
              if (x == 0)
                {
                  zmin = 0;
                }
              else
                {
                  zmin = intervals[z - 1];
                }
              pos = x + D * y + D * D * z;
              block_pos = grid_c[pos];

              for (int point = 0; point < points_block_c[pos]; point++)
                {
                  sum++;
                  if (c[(block_pos + point) * DIM + 0] >= xmax
                      || c[(block_pos + point) * DIM + 0] < xmin)
                    {
                      fails++;
                      // printf("fail at %d \n", block_pos );
                    }
                  if (c[(block_pos + point) * DIM + 1] >= ymax
                      || c[(block_pos + point) * DIM + 1] < ymin)
                    {
                      fails++;
                      // printf("fail at %d \n", block_pos );
                    }
                  if (c[(block_pos + point) * DIM + 2] >= zmax
                      || c[(block_pos + point) * DIM + 2] < zmin)
                    {
                      fails++;
                      // printf("fail at %d \n", block_pos );
                    }
                }
            }
        }
    }

  printf ("GRID VALIDATION POINTS:%d FAILS:%d\n", sum, fails);
}

void
validate_search (const float *q, const float *c, const int *closest, int N,
                 int D)
{
  float mindist, dist;
  int close;

  int fails = 0;

  for (int i = 0; i < VALIDATE; i++)
    {
      mindist = (1 << 10);
      for (int j = 0; j < N; j++)
        {
          dist = (q[i * DIM + 0] - c[j * DIM + 0])
                 * (q[i * DIM + 0] - c[j * DIM + 0]);
          dist += (q[i * DIM + 1] - c[j * DIM + 1])
                  * (q[i * DIM + 1] - c[j * DIM + 1]);
          dist += (q[i * DIM + 2] - c[j * DIM + 2])
                  * (q[i * DIM + 2] - c[j * DIM + 2]);
          dist = sqrtf (dist);
          if (dist < mindist)
            {
              close = j;
              mindist = dist;
            }
        }
      if (close != closest[i])
        {
          // printf ("intex %d %d dists %f %f  q :%f %f %f c: %f %f %f\n",
          // close,
          //  closest[i], mindist, mindists[i], q[i * DIM + 0],
          //  q[i * DIM + 1], q[i * DIM + 2], c[close * DIM + 0],
          //  c[close * DIM + 1], c[close * DIM + 2]);
          // int x, y, z;
          // x = (int) (q[i * DIM + 0] * D);
          // y = (int) (q[i * DIM + 1] * D);
          // z = (int) (q[i * DIM + 2] * D);

          // printf ("q : %d %d %d ", x, y, z);
          // x = (int) (c[close * DIM + 0] * D);
          // y = (int) (c[close * DIM + 1] * D);
          // z = (int) (c[close * DIM + 2] * D);

          // printf ("c: %d %d %d \n", x, y, z);

          fails++;
        }
    }
  printf ("SEARCH VALIDATION POINTS: %d FAILS: %d\n", VALIDATE, fails);
}

void
write_file (double time_var, const char *filename, const char *mode)
{
  FILE *fptr;
  // open the file
  char filepath[64] = "output_data/";
  strcat (filepath, filename);
  fptr = fopen (filepath, mode);
  if (!fptr)
    {
      printf ("Error: Can't open file %s", filepath);
      return;
    }
  // print the time in file
  fprintf (fptr, "%lf ", time_var);
  // close file
  fclose (fptr);
  return;
}
