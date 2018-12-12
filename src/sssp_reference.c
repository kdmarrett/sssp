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
#define TOTAL_P      8
#define MAX_EDGES    2^27
#define MAX_ITER     2^24
#define BURST_SIZE   256
#define BURST_WIDTH  8*56
#define MEM_WORD_WIDTH 56
#define INDEX_WIDTH  24
#define WEIGHT_WIDTH 8

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


void clean_shortest(data_v* dist) {
	int i;
	for(i=0;i<g.nlocalverts;i++) dist[i]=-1.0;
}

template<typename To, typename From>
inline to Reinterpret(const From& val){
  return reinterpret_cast<const To&>(val);
}

void Load(const bool enable,
          const ap_uint<BURST_WIDTH>* data_dram,
          data_i* s_local,
          data_i* d_local,
          data_v* weight_local){
#pragma HLS inline off
  if(enable){
    for(int i = 0; i < BURST_SIZE / P; i++){
    #pragma HLS pipeline
      ap_uint<BURST_WIDTH> tmp = data_dram[i];
      for(int j = 0; j < P; j++){
      #pragma HLS unroll
        s_local[i*P + j] = Reinterpret<data_i>(static_cast<unsigned>(
                            tmp((j + 1) * MEM_WORD_WIDTH - 1, (j + 1) * MEM_WORD_WIDTH - INDEX_WIDTH)));
        d_local[i*P + j] = Reinterpret<data_i>(static_cast<unsigned>(
                            tmp((j + 1) * MEM_WORD_WIDTH - INDEX_WIDTH - 1, (j + 1) * MEM_WORD_WIDTH - 2*INDEX_WIDTH)));
        weight_local[i*P + j] = Reinterpret<data_v>(static_cast<unsigned>(
                            tmp(j * MEM_WORD_WIDTH + WEIGHT_WIDTH - 1, j * MEM_WORD_WIDTH)));                    
        
      }
    }
  }
}

void initialize(data_v* dist) {
	int i;
	for(i = 0; i < g.nlocalverts; i++){
  #pragma HLS pipeline II=1
    dist[i] = -1.0;
    //pred[i] = -1;
 }
}

inline uint1 compare(int2 up1, 
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
                   data_v* dist_d,
                   int2* update_signal6, 
                   data_i* s6, 
                   data_i* d6, 
                   data_v* weights6, 
                   data_v* dist_s6,
                   data_v* dist_d6){
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
    
    data_v dist_d1[8];
    #pragma HLS array_partition variable=dist_d1 complete
    data_v dist_d2[8];
    #pragma HLS array_partition variable=dist_d2 complete
    data_v dist_d3[8];
    #pragma HLS array_partition variable=dist_d3 complete
    data_v dist_d4[8];
    #pragma HLS array_partition variable=dist_d4 complete
    data_v dist_d5[8];
    #pragma HLS array_partition variable=dist_d5 complete
    
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
      int2 up = update_signal[2*i];
      data_v w = weights[2*i];
      data_i ss = s[2*i];
      data_i dd = d[2*i];
      data_v dist_ss = dist_s[2*i];
      data_v dist_dd = dist_d[2*i];
      
      update_signal1[2*i] = update_signal[2*i+1];
      weights1[2*i] = weights[2*i+1];
      s1[2*i] = s[2*i+1];
      d1[2*i] = d[2*i+1];
      dist_s1[2*i] = dist_s[2*i+1];
      dist_d1[2*i] = dist_d[2*i+1];
      
      update_signal1[2*i+1] = up;
      weights1[2*i+1] = w;
      s1[2*i+1] = ss;
      d1[2*i+1] = dd;
      dist_s1[2*i+1] = dist_ss;
      dist_d1[2*i+1] = dist_dd;
    } else {
      update_signal1[2*i] = update_signal[2*i];
      weights1[2*i] = weights[2*i];
      s1[2*i] = s[2*i];
      d1[2*i] = d[2*i];
      dist_s1[2*i] = dist_s[2*i];
      dist_d1[2*i] = dist_d[2*i];
      
      update_signal1[2*i+1] = update_signal[2*i+1];
      weights1[2*i+1] = weights[2*i+1];
      s1[2*i+1] = s[2*i+1];
      d1[2*i+1] = d[2*i];
      dist_s1[2*i+1] = dist_s[2*i+1];
      dist_d1[2*i+1] = dist_d[2*i+1];
    }
  }
  
  //stage2
  for(int i = 0; i < 2; i++){
    #pragma HLS unroll
    if(compare(update_signal1[i], update_signal1[3-i], d1[i], d1[3-i], weights1[i]+dist_s1[3-i], weights1[i]+dist_s1[3-i]) == 1){
      int2 up = update_signal1[i];
      data_v w = weights1[i];
      data_i ss = s1[i];
      data_i dd = d1[i];
      data_v dist_ss = dist_s1[i];
      data_v dist_dd = dist_d1[i];
      
      update_signal2[i] = update_signal1[3-i];
      weights2[i] = weights1[3-i];
      s2[i] = s1[3-i];
      d2[i] = d1[3-i];
      dist_s2[i] = dist_s1[3-i];
      dist_d2[i] = dist_d1[3-i];
      
      update_signal2[3-i] = up;
      weights2[3-i] = w;
      s2[3-i] = ss;
      d2[3-i] = dd;
      dist_s2[3-i] = dist_ss;
      dist_d2[3-i] = dist_dd;
    } else {
      update_signal2[3-i] = update_signal1[3-i];
      weights2[3-i] = weights1[3-i];
      s2[3-i] = s1[3-i];
      d2[3-i] = d1[3-i];
      dist_s2[3-i] = dist_s1[3-i];
      dist_d2[3-i] = dist_d1[3-i];
      
      update_signal2[i] = update_signal1[i];
      weights2[i] = weights1[i];
      s2[i] = s1[i];
      d2[i] = d1[i];
      dist_s2[i] = dist_s1[i];
      dist_d2[i] = dist_d1[i];
    }
    
    if(compare(update_signal1[i+4], update_signal1[7-i], d1[i+4], d1[7-i], weights1[i+4]+dist_s1[7-i], weights1[i+4]+dist_s1[7-i]) == 1){
      int2 up = update_signal1[i+4];
      data_v w = weights1[i+4];
      data_i ss = s1[i+4];
      data_i dd = d1[i+4];
      data_v dist_ss = dist_s1[i+4];
      data_v dist_dd = dist_d1[i+4];
      
      update_signal2[i+4] = update_signal1[7-i];
      weights2[i+4] = weights1[7-i];
      s2[i+4] = s1[7-i];
      d2[i+4] = d1[7-i];
      dist_s2[i+4] = dist_s1[7-i];
      dist_d2[i+4] = dist_d1[7-i];
      
      update_signal2[7-i] = up;
      weights2[7-i] = w;
      s2[7-i] = ss;
      d2[7-i] = dd;
      dist_s2[7-i] = dist_ss;
      dist_d2[7-i] = dist_dd;
    } else {
      update_signal2[7-i] = update_signal1[7-i];
      weights2[7-i] = weights1[7-i];
      s2[7-i] = s1[7-i];
      d2[7-i] = d1[7-i];
      dist_s2[7-i] = dist_s1[7-i];
      dist_d2[7-i] = dist_d1[7-i];
      
      update_signal2[i+4] = update_signal1[i+4];
      weights2[i+4] = weights1[i+4];
      s2[i+4] = s1[i+4];
      d2[i+4] = d1[i+4];
      dist_s2[i+4] = dist_s1[i+4];
      dist_d2[i+4] = dist_d1[i+4];
    }
  }
  
  //stage 3
  for(int i = 0; i < 4; i++){
    #pragma HLS unroll
    if(compare(update_signal2[2*i], update_signal2[2*i+1], d2[2*i], d2[2*i+1], weights2[2*i]+dist_s2[2*i], weights2[2*i+1]+dist_s2[2*i+1]) == 1){
      int2 up = update_signal2[2*i];
      data_v w = weights2[2*i];
      data_i ss = s2[2*i];
      data_i dd = d2[2*i];
      data_v dist_ss = dist_s2[2*i];
      data_v dist_dd = dist_d2[2*i];
      
      update_signal3[2*i] = update_signal2[2*i+1];
      weights3[2*i] = weights2[2*i+1];
      s3[2*i] = s2[2*i+1];
      d3[2*i] = d2[2*i+1];
      dist_s3[2*i] = dist_s2[2*i+1];
      dist_d3[2*i] = dist_d2[2*i+1];
      
      update_signal3[2*i+1] = up;
      weights3[2*i+1] = w;
      s3[2*i+1] = ss;
      d3[2*i+1] = dd;
      dist_s3[2*i+1] = dist_ss;
      dist_d3[2*i+1] = dist_dd;
    } else {
      update_signal3[2*i] = update_signal2[2*i];
      weights3[2*i] = weights2[2*i];
      s3[2*i] = s2[2*i];
      d3[2*i] = d2[2*i];
      dist_s3[2*i] = dist_s2[2*i];
      dist_d3[2*i] = dist_d2[2*i];
      
      update_signal3[2*i+1] = update_signal2[2*i+1];
      weights3[2*i+1] = weights2[2*i+1];
      s3[2*i+1] = s2[2*i+1];
      d3[2*i+1] = d2[2*i+1];
      dist_s3[2*i+1] = dist_s2[2*i+1];
      dist_d3[2*i+1] = dist_d2[2*i+1];
    }
  }
  
  //stage 4
  for(int i = 0; i < 4; i++){
    #pragma HLS unroll
    if(compare(update_signal3[i], update_signal3[7-i], d3[i], d3[7-i], weights3[i]+dist_s3[7-i], weights3[i]+dist_s3[7-i]) == 1){
      int2 up = update_signal3[i];
      data_v w = weights3[i];
      data_i ss = s3[i];
      data_i dd = d3[i];
      data_v dist_ss = dist_s3[i];
      data_v dist_dd = dist_d3[i];
      
      update_signal4[i] = update_signal3[7-i];
      weights4[i] = weights3[7-i];
      s4[i] = s3[7-i];
      d4[i] = d3[7-i];
      dist_s4[i] = dist_s3[7-i];
      dist_d4[i] = dist_d3[7-i];
      
      update_signal4[7-i] = up;
      weights4[7-i] = w;
      s4[7-i] = ss;
      d4[7-i] = dd;
      dist_s4[7-i] = dist_ss;
      dist_d4[7-i] = dist_dd;
    }else {
      update_signal4[7-i] = update_signal3[7-i];
      weights4[7-i] = weights3[7-i];
      s4[7-i] = s3[7-i];
      d4[7-i] = d3[7-i];
      dist_s4[7-i] = dist_s3[7-i];
      dist_d4[7-i] = dist_d3[7-i];
      
      update_signal4[i] = update_signal3[i];
      weights4[i] = weights3[i];
      s4[i] = s3[i];
      d4[i] = d3[i];
      dist_s4[i] = dist_s3[i];
      dist_d4[i] = dist_d3[i];
    }
  }
  
  //stage5
  for(int i = 0; i < 2; i++){
    #pragma HLS unroll
    if(compare(update_signal4[i], update_signal4[i+2], d4[i], d4[i+2], weights4[i]+dist_s4[i+2], weights4[i]+dist_s4[i+2]) == 1){
      int2 up = update_signal4[i];
      data_v w = weights4[i];
      data_i ss = s4[i];
      data_i dd = d4[i];
      data_v dist_ss = dist_s4[i];
      data_v dist_dd = dist_d4[i];
      
      update_signal5[i] = update_signal4[i+2];
      weights5[i] = weights4[i+2];
      s5[i] = s4[i+2];
      d5[i] = d4[i+2];
      dist_s5[i] = dist_s4[i+2];
      dist_d5[i] = dist_d4[i+2];
      
      update_signal5[i+2] = up;
      weights5[i+2] = w;
      s5[i+2] = ss];
      d5[i+2] = dd;
      dist_s5[i+2] = dist_ss;
      dist_d5[i+2] = dist_dd;
    } else {
      update_signal5[i+2] = update_signal4[i+2];
      weights5[i+2] = weights4[i+2];
      s5[i+2] = s4[i+2];
      d5[i+2] = d4[i+2];
      dist_s5[i+2] = dist_s4[i+2];
      dist_d5[i+2] = dist_d4[i+2];
      
      update_signal5[i] = update_signal4[i];
      weights5[i] = weights4[i];
      s5[i] = s4[i];
      d5[i] = d4[i];
      dist_s5[i] = dist_s4[i];
      dist_d5[i] = dist_d4[i];
    }
    
    if(compare(update_signal4[i+4], update_signal4[i+6], d4[i+4], d4[i+6], weights4[i+4]+dist_s4[i+6], weights4[i+4]+dist_s4[i+6]) == 1){
      int2 up = update_signal4[i+4];
      data_v w = weights4[i+4];
      data_i ss = s4[i+4];
      data_i dd = d4[i+4];
      data_v dist_ss = dist_s4[i+4];
      data_v dist_dd = dist_d4[i+4];
      
      update_signal5[i+4] = update_signal4[i+6];
      weights5[i+4] = weights4[i+6];
      s5[i+4] = s4[i+6];
      d5[i+4] = d4[i+6];
      dist_s5[i+4] = dist_s4[i+6];
      dist_d5[i+4] = dist_d4[i+6];
      
      update_signal5[i+6] = up;
      weights5[i+6] = w;
      s5[i+6] = ss;
      d5[i+6] = dd;
      dist_s5[i+6] = dist_ss;
      dist_d5[i+6] = dist_dd;
    } else {
      update_signal5[i+6] = update_signal4[i+6];
      weights5[i+6] = weights4[i+6];
      s5[i+6] = s4[i+6];
      d5[i+6] = d4[i+6];
      dist_s5[i+6] = dist_s4[i+6];
      dist_d5[i+6] = dist_d4[i+6];
      
      update_signal5[i+4] = update_signal4[i+4];
      weights5[i+4] = weights4[i+4];
      s5[i+4] = s4[i+4];
      d5[i+4] = d4[i+4];
      dist_s5[i+4] = dist_s4[i+4];
      dist_d5[i+4] = dist_d4[i+4];
    }
     
  }
  
  //stage 6
  for(int i = 0; i < 4; i++){
    #pragma HLS unroll
    if(compare(update_signal5[2*i], update_signal5[2*i+1], d5[2*i], d5[2*i+1], weights5[2*i]+dist_s5[2*i], weights5[2*i+1]+dist_s5[2*i+1]) == 1){
      int2 up = update_signal5[2*i];
      data_v w = weights5[2*i];
      data_i ss = s5[2*i];
      data_i dd = d5[2*i];
      data_v dist_ss = dist_s5[2*i];
      data_v dist_dd = dist_d5[2*i];
      
      update_signal6[2*i] = update_signal5[2*i+1];
      weights6[2*i] = weights5[2*i+1];
      s6[2*i] = s5[2*i+1];
      d6[2*i] = d5[2*i+1];
      dist_s6[2*i] = dist_s5[2*i+1];
      dist_d6[2*i] = dist_d5[2*i+1];
      
      update_signal6[2*i+1] = up;
      weights6[2*i+1] = w;
      s6[2*i+1] = ss;
      d6[2*i+1] = dd;
      dist_s6[2*i+1] = dist_ss;
      dist_d6[2*i+1] = dist_dd;
    } else {
      update_signal6[2*i] = update_signal5[2*i];
      weights6[2*i] = weights5[2*i];
      s6[2*i] = s5[2*i];
      d6[2*i] = d5[2*i];
      dist_s6[2*i] = dist_s5[2*i];
      dist_d6[2*i] = dist_d5[2*i];
      
      update_signal6[2*i+1] = update_signal5[2*i+1];
      weights6[2*i+1] = weights5[2*i+1];
      s6[2*i+1] = s5[2*i+1];
      d6[2*i+1] = d5[2*i+1];
      dist_s6[2*i+1] = dist_s5[2*i+1];
      dist_d6[2*i+1] = dist_d5[2*i+1];
    }
  }

}

void mem_read(data_v* dist_d, 
              data_i* d, 
              int2* up_in,
              int2* up_out, 
              data_v* dist_local){
#pragma HLS inline off
// data forwarding
  for(int i = 0; i < P; i++){
    #pragma HLS unroll factor=8
    dist_d[i] = dist_local[d[i]];
  }
}

void computation_block(int2* up_in, 
                       data_v* weights, 
                       data_v* dist_s, 
                       data_v* dist_d, 
                       data_i* s, 
                       data_i* d, 
                       int2* up_out, 
                       data_v* dist_out,
                       data_i* s_out,
                       data_i* d_out){
#pragma HLS inline off
//data forwarding

  for(int i = 0; i < P; i++){
    #pragma HLS unroll factor=8
    d_out[i] = d[i];
    s_out[i] = s[i];
    data_v tmp = weights[i] + dist_s[i];
    if(tmp < dist_d[i]){
        dist_out[i] = tmp;
        if(up_in[i] == 1) up_out[i] = 1;
        else              up_out[i] = 0;
    } else {
        dist_out[i] = dist_d[i];
        up_out[i] = 0;
    }
    }
  }
}

void computation_block_revised(uint* terminate,
                       int2* up_in, 
                       data_v* weights, 
                       data_v* dist_s, 
                       data_v* dist_d, 
                       data_i* s, 
                       data_i* d,  
                       data_v* dist_global,
                       data_i* pred){
#pragma HLS inline off

  uint t[P];
  #pragma HLS array_partition variable=t complete
  for(int i = 0; i < P; i++){
    #pragma HLS unroll factor=8
    t[i] = 1;
  }
  
  for(int i = 0; i < P; i++){
    #pragma HLS unroll factor=8
    //d_out[i] = d[i];
    //s_out[i] = s[i];
    data_v tmp = weights[i] + dist_s[i];
    data_v dest = dist_global[d[i]];
    if(up_in[i] == 1){
      if(tmp < dest || dest < 0){
        if(tmp > 0){
          dist_global[d[i]] = tmp;
          pred[d[i]] = s[i];
          terminate[0] = 0;
        }
      }
    } 
    
  }
  
  if(terminate[0] == 1){
    terminate[0] = (t[0] & t[1] & t[2] & t[3] & t[4] & t[5] & t[6] & t[7]);
  }
}

void mem_write(uint* terminate,
               int2* up_in, 
               data_v* val_in,
               data_i* s_in, 
               data_i* d_in,
               int2* up_out, 
               data_v* val_out,
               data_i* s_out, 
               data_i* d_out){
#pragma HLS inline off
  uint t[P];
  #pragma HLS array_partition variable=t complete
  for(int i = 0; i < P; i++){
    #pragma HLS unroll factor=8
    t[i] = 1;
  }
  for(int i = 0; i < P; i++){
    #pragma HLS unroll factor=8
    if(up_in[i] == 1) t[i] = 0;
    up_out[i] = up_in[i];
    val_out[i] = val_in[i];
    s_out[i] = s_in[i];
    d_out[i] = d_in[i];
  }
  
  if(terminate[0] == 1){
    terminate[0] = (t[0] & t[1] & t[2] & t[3] & t[4] & t[5] & t[6] & t[7]);
  }
}

void data_forwarding_unit(data_v* val_fwd, 
                          int2* up_fwd,
                          data_i* d_fwd, 
                          data_v* val,
                          int2* up,
                          data_i* d){
  for(int i = 0; i < P; i++){
    for(int j = 0; j < P; i++){
      if(up_fwd[i] == 1 && (d_fwd[i] == d_fwd[j]) && up[j] == 1 && (val_fwd[i] < val[j])){
        up[j] = 0;
        val[j] = val_fwd[i];
      }
    }
  }
}

void computation_unit(uint* terminate, int2* update_sginal, data_i* s, data_i* d, data_v* weights, data_v* dist_s, data_v* dist_d, int2* update_signal4, data_v* update_values4, data_i* s4, data_i* d4){
#pragma HLS inline off 

  // output of sorting block
  data_v dist_s1[8];
  #pragma HLS array_partition variable=dist_s1 complete
  data_v dist_d1[8];
  #pragma HLS array_partition variable=dist_d1 complete
  data_v weights1[8];
  #pragma HLS array_partition variable=weights1 complete
  data_v s1[8];
  #pragma HLS array_partition variable=s1 complete
  data_v d1[8];
  #pragma HLS array_partition variable=d1 complete
  data_v update_signal1[8];
  #pragma HLS array_partition variable=update_signal1 complete
  
  // output of memory read block
  data_v dist_s2[8];
  #pragma HLS array_partition variable=dist_s2 complete
  data_v dist_d2[8];
  #pragma HLS array_partition variable=dist_d2 complete
  
  data_v weights2[8];
  #pragma HLS array_partition variable=weights2 complete
  data_v s2[8];
  #pragma HLS array_partition variable=s2 complete
  data_v d2[8];
  #pragma HLS array_partition variable=d2 complete
  data_v update_signal2[8];
  #pragma HLS array_partition variable=update_signal2 complete
    
#pragma HLS pipeline II=1
  int terminate = 1;
  
// sorting block
  sorting_block(update_signal, s, d, weights, dist_s, dist_d, update_signal1, s1, d1, weights1, dist_s1, dist_d1);
  
// mem read block
  mem_read(update_signal1, weights1, dist_s1, dist_d1, s1, d1, update_signal2, weights2, dist_s2, dist_d2, s2, d2);
   
// computation block
  data_v update_values3[P];
  #pragma HLS array_partition variable=update_values3 complete
  int2 update_signal3[P];
  #pragma HLS array_partition variable=update_signal3 complete
  data_v d3[8];
  #pragma HLS array_partition variable=d3 complete
  data_v s3[8];
  #pragma HLS array_partition variable=s3 complete
  computation_block(update_signal2, weights2, dist_s2, dist_d2, s2, d2, update_signal3, update_values3, s3, d3);
  
// mem write block
  mem_write(terminate, update_signal3, update_values3, s3, d3, update_signal4, update_values4, s4, d4);
  
// data forwarding
}


extern "C" {

void run_sssp(int64 nlocaledges,
              data_i root,
              data_i* pred,
              data_v* dist,
              const ap_uint<BURST_WDITH>* dram1,
              /*data_v* weight_dram2,
              data_v* weight_dram3,
              data_v* weight_dram4*/) {
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
    data_v dist_com5_s[8];
    #pragma HLS array_partition variable=dist_com5_s complete
    data_v dist_com6_s[8];
    #pragma HLS array_partition variable=dist_com6_s complete
    data_v dist_com7_s[8];
    #pragma HLS array_partition variable=dist_com7_s complete
    data_v dist_com8_s[8];
    #pragma HLS array_partition variable=dist_com8_s complete
    
    data_v dist_com1_d[8];
    #pragma HLS array_partition variable=dist_com1_d complete
    data_v dist_com2_d[8];
    #pragma HLS array_partition variable=dist_com2_d complete
    data_v dist_com3_d[8];
    #pragma HLS array_partition variable=dist_com3_d complete
    data_v dist_com4_d[8];
    #pragma HLS array_partition variable=dist_com4_d complete
    data_v dist_com5_d[8];
    #pragma HLS array_partition variable=dist_com5_d complete
    data_v dist_com6_d[8];
    #pragma HLS array_partition variable=dist_com6_d complete
    data_v dist_com7_d[8];
    #pragma HLS array_partition variable=dist_com7_d complete
    data_v dist_com8_d[8];
    #pragma HLS array_partition variable=dist_com8_d complete
    
    // source indices
    data_i com1_s[8];
    #pragma HLS array_partition variable=com1_s complete
    data_i com2_s[8];
    #pragma HLS array_partition variable=com2_s complete
    data_i com3_s[8];
    #pragma HLS array_partition variable=com3_s complete
    data_i com4_s[8];
    #pragma HLS array_partition variable=com4_s complete
    data_i com5_s[8];
    #pragma HLS array_partition variable=com5_s complete
    data_i com6_s[8];
    #pragma HLS array_partition variable=com6_s complete
    data_i com7_s[8];
    #pragma HLS array_partition variable=com7_s complete
    data_i com8_s[8];
    #pragma HLS array_partition variable=com8_s complete
    
    // destination indices
    data_i com1_d[8];
    #pragma HLS array_partition variable=com1_d complete
    data_i com2_d[8];
    #pragma HLS array_partition variable=com2_d complete
    data_i com3_d[8];
    #pragma HLS array_partition variable=com3_d complete
    data_i com4_d[8];
    #pragma HLS array_partition variable=com4_d complete
    data_i com5_d[8];
    #pragma HLS array_partition variable=com5_d complete
    data_i com6_d[8];
    #pragma HLS array_partition variable=com6_d complete
    data_i com7_d[8];
    #pragma HLS array_partition variable=com7_d complete
    data_i com8_d[8];
    #pragma HLS array_partition variable=com8_d complete
    
    // weights of edges
    data_v com1_weights[8];
    #pragma HLS array_partition variable=com1_weights complete
    data_v com2_weights[8];
    #pragma HLS array_partition variable=com2_weights complete
    data_v com3_weights[8];
    #pragma HLS array_partition variable=com3_weights complete
    data_v com4_weights[8];
    #pragma HLS array_partition variable=com4_weights complete
    data_v com5_weights[8];
    #pragma HLS array_partition variable=com5_weights complete
    data_v com6_weights[8];
    #pragma HLS array_partition variable=com6_weights complete
    data_v com7_weights[8];
    #pragma HLS array_partition variable=com7_weights complete
    data_v com8_weights[8];
    #pragma HLS array_partition variable=com8_weights complete
    
    // update signals
    int2 com1_update_signal[8];
    #pragma HLS array_partition variable=com1_update_signal complete
    int2 com2_update_signal[8];
    #pragma HLS array_partition variable=com2_update_signal complete
    int2 com3_update_signal[8];
    #pragma HLS array_partition variable=com3_update_signal complete
    int2 com4_update_signal[8];
    #pragma HLS array_partition variable=com4_update_signal complete
    int2 com5_update_signal[8];
    #pragma HLS array_partition variable=com5_update_signal complete
    int2 com6_update_signal[8];
    #pragma HLS array_partition variable=com6_update_signal complete
    int2 com7_update_signal[8];
    #pragma HLS array_partition variable=com7_update_signal complete
    int2 com8_update_signal[8];
    #pragma HLS array_partition variable=com8_update_signal complete

    // initialize : dist on URAM, no need to initialize pred
    data_v dist_local[MAX_VERTICES];
    #pragma HLS RESOURCE variable=dist_local core=XPM_MEMORY uram
    #pragma HLS array_partition variable=dist_local cyclic factor=32 // ?? check to see if you can use complete
    initialize(dist_local);
    dist_local[root]=0.0;
    pred[root]=root; 
    
    const int min_trip_count = 1;
    const int max_trip_count = min_trip_count + MAX_ITER / BURST_SIZE;
    
    data_i s_local1[BURST_SIZE];
    #pragma HLS array_partition variable=s_local1 complete
    data_i d_local1[BURST_SIZE];
    #pragma HLS array_partition variable=d_local1 complete
    data_v weight_local1[BURST_SIZE];
    #pragma HLS array_partition variable=weight_local1 complete
    
    data_i s_local2[BURST_SIZE];
    #pragma HLS array_partition variable=s_local2 complete
    data_i d_local2[BURST_SIZE];
    #pragma HLS array_partition variable=d_local2 complete
    data_v weight_local2[BURST_SIZE];
    #pragma HLS array_partition variable=weight_local2 complete
    
    // loop over iteration
    for(i = 0; i < MAX_VERTICES; i++){
      if( i < g.nlocalvertices-1 ){
      uint1 terminate[2];
      #pragma HLS array_partition variable=terminate complete
      for(int ii = 0; ii < 2; ii++) terminate[ii] = 1;
      
    // loop over edges/(computation_unit*p)
      for(j = 0; j < MAX_ITER + BURST_SIZE; j+=BURST_SIZE, 
                                dram1 += BURST_SIZE / P){
        #pragma HLS loop_tripcount min = min_trip_count max = max_trip_count
        if(j < nlocaledges + BURST_SIZE){
          if((j / BURST_SIZE) % 2)){
            Load( j < nlocaledges, dram1, s_local1, d_local1, weight_local1);
            for(l = 0; l < BURST_SIZE / P; l++, s_local2+=P,
                              d_local2+=P, weight_local2+=P){
              if((l / (BURST_SIZE / P) %2)){
                  for(int ii = 0; ii < P; ii++){
                    #pragma HLS unroll factor=8
                    dist_com3_s[ii] = dist_local[s_local2[ii]];
                    com3_update_signal[ii] = 1;
                  }
                  sorting_block(com3_update_signal, s_local2, d_local2, weight_local2, dist_com3_s, dist_com3_d, com4_update_signal, com4_s, com4_d, com4_weights, dist_com4_s, dist_com4_d);
                  computation_block_revised(terminate+1, com4_update_signal,  com4_weights, dist_com4_s, dist_com4_d, com4_s, com4_d, dist_local, pred);
                  
              } else {
                  for(int ii = 0; ii < P; ii++){
                    #pragma HLS unroll factor=8
                    dist_com1_s[ii] = dist_local[s_local2[ii]];
                    com1_update_signal[ii] = 1;
                  }
                  sorting_block(com1_update_signal, s_local2, d_local2, weight_local2, dist_com1_s, dist_com1_d, com2_update_signal, com2_s, com2_d, com2_weights, dist_com2_s, dist_com2_d);
                  computation_block_revised(terminate, com2_update_signal,  com2_weights, dist_com2_s, dist_com2_d, com2_s, com2_d, dist_local, pred);
              }
            }
          } else {
            Load( j < nlocaledges, dram1, s_local2, d_local2, weight_local2);
            for(l = 0; l < BURST_SIZE / P; l++, s_local1+=P,
                              d_local1+=P, weight_local1+=P){
              if((l / (BURST_SIZE / P) %2)){
                  for(int ii = 0; ii < P; ii++){
                    #pragma HLS unroll factor=8
                    dist_com7_s[ii] = dist_local[s_local1[ii]];
                    com7_update_signal[ii] = 1;
                  }
                  sorting_block(com7_update_signal, s_local1, d_local1, weight_local1, dist_com7_s, dist_com7_d, com8_update_signal, com8_s, com8_d, com8_weights, dist_com8_s, dist_com8_d);
                  computation_block_revised(terminate+2, com8_update_signal,  com8_weights, dist_com8_s, dist_com8_d, com8_s, com8_d, dist_local, pred);
                  
              } else {
                  for(int ii = 0; ii < P; ii++){
                    #pragma HLS unroll factor=8
                    dist_com5_s[ii] = dist_local[s_local1[ii]];
                    com5_update_signal[ii] = 1;
                  }
                  sorting_block(com5_update_signal, s_local1, d_local1, weight_local1, dist_com5_s, dist_com5_d, com6_update_signal, com6_s, com6_d, com6_weights, dist_com6_s, dist_com6_d);
                  computation_block_revised(terminate+3, com6_update_signal,  com6_weights, dist_com6_s, dist_com6_d, com6_s, com6_d, dist_local, pred);
              }
            }
          }
        }
        
      
      }
      if(terminate[0] == 1) break;
      }
    }
    
    // write vertices weights back to DRAM
    for(i = 0; i < MAX_VERTICES; i++){
    #pragma HLS pipeline
      if(i < g.nlocalvertices) dist[i] = dist_local[i];
    }

    

} // extern "C"


#endif
