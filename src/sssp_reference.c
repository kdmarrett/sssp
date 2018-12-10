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

typedef ap_fixed<8, 2> data_v; // representing weight of vertices and edges
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

void mem_read(data_v* dist_d, 
              data_i* d, 
              int2* update_signal, 
              data_v* dist_local){
#pragma HLS inline off
// data forwarding
  for(int i = 0; i < P; i++){
    #pragma HLS unroll factor=8
    dist_d[i] = dist_local[d[i]];
  }
}

uint1 compare(int2 up1, 
              int2 up2, 
              data_i ind1, 
              data_i ind2, 
              data_v val1, 
              data_v val2){
  uint swap = 0;
  if(up1 == 0 & up2 == 1) swap = 1;
  else if (up1 == 1 & up2 == 1 & (ind2 < ind1)) swap = 1;
  else if (up1 == 1 & up2 == 1 & (ind2 == ind1) & val2 < val1) swap = 1;
  else swap = 0;
  
  return swap;
}

void sorting_block(int2* update_signal, 
                   data_i* s, 
                   data_i* d, 
                   data_v* weights, 
                   data_v* dist_s,
                   int2* update_signal6, 
                   data_i* s6, 
                   data_i* d6, 
                   data_v* weights6, 
                   data_v* dist_s6){
#pragma HLS inline off

    data_v dist_s1[8];
    #pragma HLS array_partition variable=dist_s1 complete
    data_v dist_s2[8];
    #pragma HLS array_partition variable=dist_s2 complete
    data_v dist_s3[8];
    #pragma HLS array_partition variable=dist_s3 complete
    data_v dist_s4[8];
    #pragma HLS array_partition variable=dist_s4 complete
    data_v dist_s5[8];
    #pragma HLS array_partition variable=dist_s5 complete
    
    // source indices
    data_i s1[8];
    #pragma HLS array_partition variable=s1 complete
    data_i s2[8];
    #pragma HLS array_partition variable=s2 complete
    data_i s3[8];
    #pragma HLS array_partition variable=s3 complete
    data_i s4[8];
    #pragma HLS array_partition variable=s4 complete
    data_i s5[8];
    #pragma HLS array_partition variable=s5 complete
    
    // destination indices
    data_i d1[8];
    #pragma HLS array_partition variable=d1 complete
    data_i d2[8];
    #pragma HLS array_partition variable=d2 complete
    data_i d3[8];
    #pragma HLS array_partition variable=d3 complete
    data_i d4[8];
    #pragma HLS array_partition variable=d4 complete
    data_i d5[8];
    #pragma HLS array_partition variable=d5 complete
    
    // weights of edges
    data_v weights1[8];
    #pragma HLS array_partition variable=weights1 complete
    data_v weights2[8];
    #pragma HLS array_partition variable=weights2 complete
    data_v weights3[8];
    #pragma HLS array_partition variable=weights3 complete
    data_v weights4[8];
    #pragma HLS array_partition variable=weights4 complete
    data_v weights5[8];
    #pragma HLS array_partition variable=weights5 complete
    
    // update signals
    int2 update_signal1[8];
    #pragma HLS array_partition variable=update_signal1 complete
    int2 update_signal2[8];
    #pragma HLS array_partition variable=update_signal2 complete
    int2 update_signal3[8];
    #pragma HLS array_partition variable=update_signal3 complete
    int2 update_signal4[8];
    #pragma HLS array_partition variable=update_signal4 complete
    int2 update_signal5[8];
    #pragma HLS array_partition variable=update_signal5 complete
    
#pragma HLS pipeline II=1
  
  //stage 1
  for(int i = 0; i < 4; i++){
    #pragma HLS unroll
    if(compare(update_signal[2*i], update_signal[2*i+1], d[2*i], d[2*i+1], weights[2*i]+dist_s[2*i], weights[2*i+1]+dist_s[2*i+1]) == 1){
      update_signal1[2*i] = update_signal[2*i+1];
      weights1[2*i] = weights[2*i+1];
      s1[2*i] = s[2*i+1];
      d1[2*i] = d[2*i+1];
      dist_s1[2*i] = dist_s[2*i+1];
      
      update_signal1[2*i+1] = update_signal[2*i];
      weights1[2*i+1] = weights[2*i];
      s1[2*i+1] = s[2*i];
      d1[2*i+1] = d[2*i];
      dist_s1[2*i+1] = dist_s[2*i];
    } else {
      update_signal1[2*i] = update_signal[2*i];
      weights1[2*i] = weights[2*i];
      s1[2*i] = s[2*i];
      d1[2*i] = d[2*i];
      dist_s1[2*i] = dist_s[2*i];
      
      update_signal1[2*i+1] = update_signal[2*i+1];
      weights1[2*i+1] = weights[2*i+1];
      s1[2*i+1] = s[2*i+1];
      d1[2*i+1] = d[2*i];
      dist_s1[2*i+1] = dist_s[2*i+1];
    }
  }
  
  //stage2
  for(int i = 0; i < 2; i++){
    #pragma HLS unroll
    if(compare(update_signal1[i], update_signal1[3-i], d1[i], d1[3-i], weights1[i]+dist_s1[3-i], weights1[i]+dist_s1[3-i]) == 1){
      update_signal2[i] = update_signal1[3-i];
      weights2[i] = weights1[3-i];
      s2[i] = s1[3-i];
      d2[i] = d1[3-i];
      dist_s2[i] = dist_s1[3-i];
      
      update_signal2[3-i] = update_signal1[i];
      weights2[3-i] = weights1[i];
      s2[3-i] = s1[i];
      d2[3-i] = d1[i];
      dist_s2[3-i] = dist_s1[i];
    } else {
      update_signal2[3-i] = update_signal1[3-i];
      weights2[3-i] = weights1[3-i];
      s2[3-i] = s1[3-i];
      d2[3-i] = d1[3-i];
      dist_s2[3-i] = dist_s1[3-i];
      
      update_signal2[i] = update_signal1[i];
      weights2[i] = weights1[i];
      s2[i] = s1[i];
      d2[i] = d1[i];
      dist_s2[i] = dist_s1[i];
    }
    
    if(compare(update_signal1[i+4], update_signal1[7-i], d1[i+4], d1[7-i], weights1[i+4]+dist_s1[7-i], weights1[i+4]+dist_s1[7-i]) == 1){
      update_signal2[i+4] = update_signal1[7-i];
      weights2[i+4] = weights1[7-i];
      s2[i+4] = s1[7-i];
      d2[i+4] = d1[7-i];
      dist_s2[i+4] = dist_s1[7-i];
      
      update_signal2[7-i] = update_signal1[i+4];
      weights2[7-i] = weights1[i+4];
      s2[7-i] = s1[i+4];
      d2[7-i] = d1[i+4];
      dist_s2[7-i] = dist_s1[i+4];
    } else {
      update_signal2[7-i] = update_signal1[7-i];
      weights2[7-i] = weights1[7-i];
      s2[7-i] = s1[7-i];
      d2[7-i] = d1[7-i];
      dist_s2[7-i] = dist_s1[7-i];
      
      update_signal2[i+4] = update_signal1[i+4];
      weights2[i+4] = weights1[i+4];
      s2[i+4] = s1[i+4];
      d2[i+4] = d1[i+4];
      dist_s2[i+4] = dist_s1[i+4];
    }
  }
  
  //stage 3
  for(int i = 0; i < 4; i++){
    #pragma HLS unroll
    if(compare(update_signal2[2*i], update_signal2[2*i+1], d2[2*i], d2[2*i+1], weights2[2*i]+dist_s2[2*i], weights2[2*i+1]+dist_s2[2*i+1]) == 1){
      update_signal3[2*i] = update_signal2[2*i+1];
      weights3[2*i] = weights2[2*i+1];
      s3[2*i] = s2[2*i+1];
      d3[2*i] = d2[2*i+1];
      dist_s3[2*i] = dist_s2[2*i+1];
      
      update_signal3[2*i+1] = update_signal2[2*i];
      weights3[2*i+1] = weights2[2*i];
      s3[2*i+1] = s2[2*i];
      d3[2*i+1] = d2[2*i];
      dist_s3[2*i+1] = dist_s2[2*i];
    } else {
      update_signal3[2*i] = update_signal2[2*i];
      weights3[2*i] = weights2[2*i];
      s3[2*i] = s2[2*i];
      d3[2*i] = d2[2*i];
      dist_s3[2*i] = dist_s2[2*i];
      
      update_signal3[2*i+1] = update_signal2[2*i+1];
      weights3[2*i+1] = weights2[2*i+1];
      s3[2*i+1] = s2[2*i+1];
      d3[2*i+1] = d2[2*i+1];
      dist_s3[2*i+1] = dist_s2[2*i+1];
    }
  }
  
  //stage 4
  for(int i = 0; i < 4; i++){
    #pragma HLS unroll
    if(compare(update_signal3[i], update_signal3[7-i], d3[i], d3[7-i], weights3[i]+dist_s3[7-i], weights3[i]+dist_s3[7-i]) == 1){
      update_signal4[i] = update_signal3[7-i];
      weights4[i] = weights3[7-i];
      s4[i] = s3[7-i];
      d4[i] = d3[7-i];
      dist_s4[i] = dist_s3[7-i];
      
      update_signal4[7-i] = update_signal3[i];
      weights4[7-i] = weights3[i];
      s4[7-i] = s3[i];
      d4[7-i] = d3[i];
      dist_s4[7-i] = dist_s3[i];
    }else {
      update_signal4[7-i] = update_signal3[7-i];
      weights4[7-i] = weights3[7-i];
      s4[7-i] = s3[7-i];
      d4[7-i] = d3[7-i];
      dist_s4[7-i] = dist_s3[7-i];
      
      update_signal4[i] = update_signal3[i];
      weights4[i] = weights3[i];
      s4[i] = s3[i];
      d4[i] = d3[i];
      dist_s4[i] = dist_s3[i];
    }
  }
  
  //stage5
  for(int i = 0; i < 2; i++){
    #pragma HLS unroll
    if(compare(update_signal4[i], update_signal4[i+2], d4[i], d4[i+2], weights4[i]+dist_s4[i+2], weights4[i]+dist_s4[i+2]) == 1){
      update_signal5[i] = update_signal4[i+2];
      weights5[i] = weights4[i+2];
      s5[i] = s4[i+2];
      d5[i] = d4[i+2];
      dist_s5[i] = dist_s4[i+2];
      
      update_signal5[i+2] = update_signal4[i];
      weights5[i+2] = weights4[i];
      s5[i+2] = s4[i];
      d5[i+2] = d4[i];
      dist_s5[i+2] = dist_s4[i];
    } else {
      update_signal5[i+2] = update_signal4[i+2];
      weights5[i+2] = weights4[i+2];
      s5[i+2] = s4[i+2];
      d5[i+2] = d4[i+2];
      dist_s5[i+2] = dist_s4[i+2];
      
      update_signal5[i] = update_signal4[i];
      weights5[i] = weights4[i];
      s5[i] = s4[i];
      d5[i] = d4[i];
      dist_s5[i] = dist_s4[i];
    }
    
    if(compare(update_signal4[i+4], update_signal4[i+6], d4[i+4], d4[i+6], weights4[i+4]+dist_s4[i+6], weights4[i+4]+dist_s4[i+6]) == 1){
      update_signal5[i+4] = update_signal4[i+6];
      weights5[i+4] = weights4[i+6];
      s5[i+4] = s4[i+6];
      d5[i+4] = d4[i+6];
      dist_s5[i+4] = dist_s4[i+6];
      
      update_signal5[i+6] = update_signal4[i+4];
      weights5[i+6] = weights4[i+4];
      s5[i+6] = s4[i+4];
      d5[i+6] = d4[i+4];
      dist_s5[i+6] = dist_s4[i+4];
    } else {
      update_signal5[i+6] = update_signal4[i+6];
      weights5[i+6] = weights4[i+6];
      s5[i+6] = s4[i+6];
      d5[i+6] = d4[i+6];
      dist_s5[i+6] = dist_s4[i+6];
      
      update_signal5[i+4] = update_signal4[i+4];
      weights5[i+4] = weights4[i+4];
      s5[i+4] = s4[i+4];
      d5[i+4] = d4[i+4];
      dist_s5[i+4] = dist_s4[i+4];
    }
     
  }
  
  //stage 6
  for(int i = 0; i < 4; i++){
    #pragma HLS unroll
    if(compare(update_signal5[2*i], update_signal5[2*i+1], d5[2*i], d5[2*i+1], weights5[2*i]+dist_s5[2*i], weights5[2*i+1]+dist_s5[2*i+1]) == 1){
      update_signal6[2*i] = update_signal5[2*i+1];
      weights6[2*i] = weights5[2*i+1];
      s6[2*i] = s5[2*i+1];
      d6[2*i] = d5[2*i+1];
      dist_s6[2*i] = dist_s5[2*i+1];
      
      update_signal6[2*i+1] = update_signal5[2*i];
      weights6[2*i+1] = weights5[2*i];
      s6[2*i+1] = s5[2*i];
      d6[2*i+1] = d5[2*i];
      dist_s6[2*i+1] = dist_s5[2*i];
    } else {
      update_signal6[2*i] = update_signal5[2*i];
      weights6[2*i] = weights5[2*i];
      s6[2*i] = s5[2*i];
      d6[2*i] = d5[2*i];
      dist_s6[2*i] = dist_s5[2*i];
      
      update_signal6[2*i+1] = update_signal5[2*i+1];
      weights6[2*i+1] = weights5[2*i+1];
      s6[2*i+1] = s5[2*i+1];
      d6[2*i+1] = d5[2*i+1];
      dist_s6[2*i+1] = dist_s5[2*i+1];
    }
  }

}

int communication_unit(int2* update_sginal, data_i* s, data_i* d, data_v* weights, data_v* dist_s, data_v* dist_local){
#pragma HLS inline off 
  data_v dist_s1[8];
  #pragma HLS array_partition variable=dist_s1 complete
    data_v weights1[8];
    #pragma HLS array_partition variable=weights1 complete
    data_v s1[8];
    #pragma HLS array_partition variable=s1 complete
    data_v d1[8];
    #pragma HLS array_partition variable=d1 complete
    data_v update_signal1[8];
    #pragma HLS array_partition variable=update_signal1 complete
    
#pragma HLS pipeline II=1
  int terminate = 1;
  
// sorting block
  sorting_block(update_signal, s, d, weights, dist_s, update_signal1, s1, d1, weights6, dist_s1)
  
// mem read block
  data_v dist_d1[P];
  #pragma HLS array_partition variable=dist_d1 complete
  mem_read(dist_d1, d1, dist_local);
   
// computation block
// data forwarding
  return terminate;
}


extern "C" {

void run_sssp(data_i root,
              data_i* pred,
              data_v* dist,
              data_v* weight_dram1,
              data_v* weight_dram2,
              data_v* weight_dram3,
              data_v* weight_dram4) {
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
    data_v com1_weights[8];
    #pragma HLS array_partition variable=com1_weights complete
    data_v com2_weights[8];
    #pragma HLS array_partition variable=com2_weights complete
    data_v com3_weights[8];
    #pragma HLS array_partition variable=com3_weights complete
    data_v com4_weights[8];
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

} // extern "C"


#endif
