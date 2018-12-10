/* Copyright (c) 2011-2017 Graph500 Steering Committee
   All rights reserved.
   Developed by:                Anton Korzh anton@korzh.us
                                Graph500 Steering Committee
                                http://www.graph500.org
   New code under University of Illinois/NCSA Open Source License
   see license.txt or https://opensource.org/licenses/NCSA
*/

// Graph500: Kernel 3 SSSP

#include "common.h"
#include "csr_reference.h"
#include "bitmap_reference.h"
#include "stdio.h"

// modification:
#include "ap_fixed.h"
#include "ap_int.h"

typedef ap_fixed<8, 2> data_v; // representing weight of vertices
typedef ap_fixed<8, 2> data_e; // representing weight of edges
typedef int24          data_i; // representing indices

#define MAX_VERTICES 2^23
#define P            8

// end of modification

// variables shared from bfs_reference
extern oned_csr_graph g;
extern int qc,q2c;
extern int* q1,*q2;
extern int* rowstarts;
extern int64_t* column,*pred_glob,visited_size;
extern unsigned long * visited;
#ifdef SSSP
//global variables as those accesed by relax handler
float *glob_dist;
float glob_maxdelta, glob_mindelta; //range for current bucket
float *weights;
volatile int lightphase;

/*
//Relaxation data type
typedef struct  __attribute__((__packed__)) relaxmsg {
	float w; //weight of an edge
	int dest_vloc; //local index of destination vertex
	int src_vloc; //local index of source vertex
} relaxmsg;

// message handler for relaxation
void relaxhndl(const relaxmsg *m) {
	int vloc = m->dest_vloc;
	float w = m->w;
	float *dest_dist = &glob_dist[vloc];
	//check if relaxation is needed: either new path is shorter or vertex not reached earlier
	if (*dest_dist < 0 || *dest_dist > w) {
		*dest_dist = w; //update distance
		pred_glob[vloc]=m->src_vloc; //update path
    //printf("Source = %d, Dest = %d, Weight = %f\n", m->src_vloc, vloc, w);

		if(lightphase && !TEST_VISITEDLOC(vloc)) //Bitmap used to track if was already relaxed with light edge
		{
			if(w < glob_maxdelta) { //if falls into current bucket needs further reprocessing
				q2[q2c++] = vloc;
				SET_VISITEDLOC(vloc);
			}
		}
   
	}
}

//Sending relaxation message
void send_relax(int64_t glob, float weight,int fromloc) {
	relaxmsg m = {weight,glob,fromloc};
    relaxhndl(&m);
}
*/

void clean_shortest(data_v* dist) {
	int i;
	for(i=0;i<g.nlocalverts;i++) dist[i]=-1.0;
}

void initialize(data_v* dist) {
	int i;
	for(i = 0; i < g.nlocalverts; i++){
  #pragma HLS pipeline II=1
    dist[i] = -1.0;
    //pred[i] = -1;
 }
}

void data_forwarding_unit(data_v* cur, int2* update signal, data_v* fwd){
  
}

void mem_read(data_v* dist_d, data_i* d, int2* update_signal, data_v* dist_local){
#pragma HLS inline off
// data forwarding
  for(int i = 0; i < P; i++){
    #pragma HLS unroll factor=8
    dist_d[i] = dist_local[d[i]];
  }
}

int communication_unit(int2* update_sginal, data_i* s, data_i* d, data_e* weights, data_v* dist_s, data_v* dist_local){
#pragma HLS inline off 
#pragma HLS pipeline II=1
  int terminate = 1;
// sorting block
// mem read block
  data_v dist_d[P];
  #pragma HLS array_partition variable=dist_d complete
  mem_read(dist_d, d, dist_local); 
// computation block
// data forwarding
  return terminate;
}


//extern "C" {

void run_sssp(data_i root,
              data_i* pred,
              data_v* dist,
              data_e* weight_dram1,
              data_e* weight_dram2,
              data_e* weight_dram3,
              data_e* weight_dram4) {
    // TODO: your modification here

	unsigned int i,j, l;
	long sum=0;

	//weights=g.weights; // size 4 * nlocaledges
	glob_dist=dist;
	pred_glob=pred;

    // arrays for each of the computation units
    // accessing source is random, so use seperate arrays for them on BRAM
    // but accessing destination can be from URAM, since it is sorted
    data_v dist_com1_s[8];
    #pragma HLS array_partition variable=dist_com1_s complete
    data_v dist_com2_s[8];
    #pragma HLS array_partition variable=dist_com2_s complete
    data_v dist_com3_s[8];
    #pragma HLS array_partition variable=dist_com3_s complete
    data_v dist_com4_s[8];
    #pragma HLS array_partition variable=dist_com4_s complete
    
    // source indices
    data_i com1_s[8];
    #pragma HLS array_partition variable=com1_s complete
    data_i com2_s[8];
    #pragma HLS array_partition variable=com2_s complete
    data_i com3_s[8];
    #pragma HLS array_partition variable=com3_s complete
    data_i com4_s[8];
    #pragma HLS array_partition variable=com4_s complete
    
    // destination indices
    data_i com1_d[8];
    #pragma HLS array_partition variable=com1_d complete
    data_i com2_d[8];
    #pragma HLS array_partition variable=com2_d complete
    data_i com3_d[8];
    #pragma HLS array_partition variable=com3_d complete
    data_i com4_d[8];
    #pragma HLS array_partition variable=com4_d complete
    
    // weights of edges
    data_e com1_weights[8];
    #pragma HLS array_partition variable=com1_weights complete
    data_e com2_weights[8];
    #pragma HLS array_partition variable=com2_weights complete
    data_e com3_weights[8];
    #pragma HLS array_partition variable=com3_weights complete
    data_e com4_weights[8];
    #pragma HLS array_partition variable=com4_weights complete
    
    // update signals
    int2 com1_update_signal[8];
    #pragma HLS array_partition variable=com1_update_signal complete
    int2 com2_update_signal[8];
    #pragma HLS array_partition variable=com2_update_signal complete
    int2 com3_update_signal[8];
    #pragma HLS array_partition variable=com3_update_signal complete
    int2 com4_update_signal[8];
    #pragma HLS array_partition variable=com4_update_signal complete

    // initialize : dist on URAM, no need to initialize pred
    data_v dist_local[MAX_VERTICES];
    #pragma HLS RESOURCE variable=dist_local core=XPM_MEMORY uram
    #pragma HLS array_partition variable=dist_local block factor=4 // ?? check to see if you can use complete
    initialize(dist_local);
    dist_local[root]=0.0;
    pred[root]=root; 
    
    // loop over iteration
    for(i = 0; i < g.nlocalverts-1; i++){
    
      int terminate = 1, terminate_com1 = 1, terminate_com2 = 1, terminate_com3 = 1, terminate_com4 = 1;
    // loop over edges/(computation_unit*p)
      for(j = 0; j < g.nlocaledges/(4*p); j++){
      #pragma HLS unroll factor = 4
    // 4 reading from dram: pack seperately as i, j, w(i,j)
    // initialize update signals
    
    // 4 computation unit
    // in each: sorting, mem read from URAM, computation unit, mem write to URAM and dram(for pred)
          communication_unit(terminate_com1, com1_update_signal, com1_s, com1_d, com1_weights, dist_com1_s, dist_local);
          communication_unit(terminate_com2, com2_update_signal, com2_s, com2_d, com2_weights, dist_com2_s, dist_local);
          communication_unit(terminate_com3, com3_update_signal, com3_s, com3_d, com3_weights, dist_com3_s, dist_local);
          communication_unit(terminate_com4, com4_update_signal, com4_s, com4_d, com4_weights, dist_com4_s, dist_local);
          
    // update pred based on update signals
      
      }
      if(terminate_com1 == 1 & terminate_com2 == 1 & terminate_com3 == 1 & terminate_com4 == 1) break;
    }
    
    // write vertices weights back to DRAM
    for(i = 0; i < g.nlocalverts; i++){
    #pragma HLS pipeline
      dist[i] = dist_local[i];
    }

    

    //printf("Starting BF nlocalverts=%d\n", g.nlocalverts);
    // iterations
	for(i = 0; i < g.nlocalverts-1; i++){
        int terminate = 1;
        // loop all edges
        for(j = 0; j < g.nlocalverts; j++){
          for(l = rowstarts[j]; l < rowstarts[j+1]; l++){
            if(dist[COLUMN(l)] < 0 || weights[l] + dist[j] < dist[COLUMN(l)]){
              if(weights[l] + dist[j] > 0){
                dist[COLUMN(l)] = weights[l] + dist[j];
                pred[COLUMN(l)] = j;
                terminate = 0;
              }
            }
          }
        }
        //printf("Checking terminate, iteration %d\n", i);
        if(terminate == 1) break;
    }
    
/*
  sum=1;
	int64_t lastvisited=1;
	while(sum!=0) {
		//1. iterate over light edges
		while(sum!=0) {
			CLEAN_VISITED();
			lightphase=1;
			for(i=0;i<qc;i++)
				for(j=rowstarts[q1[i]];j<rowstarts[q1[i]+1];j++)
					if(weights[j]<delta)
						send_relax(COLUMN(j),dist[q1[i]]+weights[j],q1[i]);

			qc=q2c;q2c=0;int *tmp=q1;q1=q2;q2=tmp;
			sum=qc;
		}
		lightphase=0;

		//2. iterate over S and heavy edges
		for(i=0;i<g.nlocalverts;i++)
			if(dist[i]>=glob_mindelta && dist[i] < glob_maxdelta) {
				for(j=rowstarts[i];j<rowstarts[i+1];j++)
					if(weights[j]>=delta)
						send_relax(COLUMN(j),dist[i]+weights[j],i);
			}

		glob_mindelta=glob_maxdelta;
		glob_maxdelta+=delta;
		qc=0;sum=0;

		//3. Bucket processing and checking termination condition
		int64_t lvlvisited=0;
		for(i=0;i<g.nlocalverts;i++)
			if(dist[i]>=glob_mindelta) {
				sum++; //how many are still to be processed
				if (dist[i] < glob_maxdelta)
					q1[qc++]=i; //this is lowest bucket
			} else if(dist[i]!=-1.0) lvlvisited++;
	}
*/
}

// } // extern "C"


#endif
