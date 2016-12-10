//====== Graph Benchmark Suites ======//
//======== Degree Centrality =========//

#include "common.h"
#include "def.h"
#include "openG.h"
#include <math.h>
#include <stack>
#include <iomanip>
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <iostream>

#ifdef SIM
#include "SIM.h"
#endif
#ifdef HMC
#include "HMC.h"
#endif

using namespace std;

unsigned itercnt = 0;
size_t beginiter = 0;
size_t enditer = 0;

class vertex_property
{
public:
    vertex_property():pr(0.1),old_pr(0.1){}

    float pr;
    float old_pr;
};
class edge_property
{
public:
    edge_property():weight(0){}
    edge_property(float x):weight(x){}

    float weight;
};

typedef openG::extGraph<vertex_property, edge_property> graph_t;
typedef graph_t::vertex_iterator    vertex_iterator;
typedef graph_t::edge_iterator      edge_iterator;

//==============================================================//
void arg_init(argument_parser & arg)
{
    arg.add_arg("damp","0.85","damping factor");
    arg.add_arg("maxiter","100","maximum allowed iteration number");
    arg.add_arg("quad","0.001","quadratic error value");
}
//==============================================================//
void init_pagerank(graph_t& g, double damp, unsigned threadnum)
{
    uint64_t chunk = (unsigned)ceil(g.num_vertices()/(double)threadnum);
    //#pragma omp parallel num_threads(threadnum)
    //{
        unsigned tid = omp_get_thread_num();

        unsigned start = tid*chunk;
        unsigned end = start + chunk;
        if (end > g.num_vertices()) end = g.num_vertices();
        
        for (unsigned vid=start;vid<end;vid++)
        {
            vertex_iterator vit = g.find_vertex(vid);
            if (vit == g.vertices_end()) continue;

            vit->property().pr = 1.0/(double)g.num_vertices();

            float weight = (vit->edges_size()>0) ? 1.0/(double)vit->edges_size() : 0;





            //int counttest = 0;

            for (edge_iterator eit=vit->edges_begin(); eit!=vit->edges_end(); eit++)
            {
            	//printf("eit: %d\n", (int)eit);
            	//counttest++;
                eit->property().weight = weight * damp + (1.0-damp)/(double)g.num_vertices();
            }
            //printf("counttest: %d\n", counttest);
        }
    //}
}
void parallel_pagerank(graph_t& g, 
        unsigned threadnum, 
        double damp,
        double quad,
        size_t maxiter,
        gBenchPerf_multi & perf, 
        int perf_group)
{
    uint64_t chunk = (unsigned)ceil(g.num_vertices()/(double)threadnum);
    //vector<float> e_vec(threadnum, 0);
    float e_vec_sum = 0;
    float random_weight = (1.0 - damp) / (double)g.num_vertices();
    bool stop = false;
    #pragma omp parallel num_threads(threadnum)
    {
        unsigned tid = omp_get_thread_num();
        //printf("tid: %u\n", tid);

        perf.open(tid, perf_group);
        perf.start(tid, perf_group); 
       
        unsigned start = tid*chunk;
        unsigned end = start + chunk;
        if (end > g.num_vertices()) end = g.num_vertices();
#ifdef SIM
        unsigned iter = 0;
#endif 

        vertex_iterator vit;

        
        __attribute__((aligned(16))) float *pr, *old_pr, *result;
        pr = (float *)_mm_malloc(4 * sizeof(float), 16);
        old_pr = (float *)_mm_malloc(4 * sizeof(float), 16);
        result = (float *)_mm_malloc(4 * sizeof(float), 16);
        
        assert(pr && old_pr && result);

        memset(pr, 0, 4 * sizeof(float));
        memset(old_pr, 0, 4 * sizeof(float));
        memset(result, 0, 4 * sizeof(float));

        __m128 pr_vec, old_pr_vec, d_vec;
        __m128 zero_vec = _mm_setzero_ps();
        vertex_iterator vit1, vit2, vit3, vit4;
        //float pr1, pr2, pr3, pr4;
        //float old_pr1, old_pr2, old_pr3, old_pr4;
        


        while(stop == false)
        {
            // Reference: PageRank Algorithm on wiki
            // PR_i = random_weigh + d * sigma(old_PR_j / L_j)

            for (unsigned vid=start;vid<end;vid++)
            {
                vertex_iterator vit = g.find_vertex(vid);
                vit->property().old_pr = vit->property().pr;
                vit->property().pr = random_weight;
            }
            #pragma omp barrier
#ifdef SIM
            SIM_BEGIN(iter==beginiter);
            iter++;
#endif
            // Push own pr to neighbour vertices
            //  can also be changed to pull based model
            //      pull based model can avoid atomic inst, 
            //      but requires predecessor list
            for (unsigned vid=start;vid<end;vid++)
            {
                vertex_iterator vit = g.find_vertex(vid);
                float pr_push = damp * vit->property().old_pr / (double) vit->edges_size();
                for (edge_iterator eit=vit->edges_begin(); eit!=vit->edges_end(); eit++)
                {
                    uint64_t dest = eit->target();
                    vertex_iterator dvit = g.find_vertex(dest);
#ifdef HMC
                    HMC_FP_ADD(&(dvit->property().pr), pr_push);
#else
                    #pragma omp atomic
                    dvit->property().pr += pr_push;
#endif
                }
            }
#ifdef SIM
            SIM_END(iter==enditer);
#endif
            // check stop condition
            #pragma omp barrier
            //e_vec[tid] = 0;
            e_vec_sum = 0;

            //cout<<"start: "<<start<<"\n";
            //cout<<"end  : "<<end<<"\n";
            //cout<<"sizeof(unsigned): "<<sizeof(unsigned)<<"\n";

            
            
            for (unsigned vid=start;vid<end;vid+=4)
            {
            	
                vit1 = g.find_vertex(vid);
                vit2 = g.find_vertex(vid + 1);
                vit3 = g.find_vertex(vid + 2);
                vit4 = g.find_vertex(vid + 3);

                pr[0] = vit1->property().pr;   old_pr[0] = vit1->property().old_pr;
                pr[1] = vit2->property().pr;   old_pr[1] = vit2->property().old_pr;
                pr[2] = vit3->property().pr;   old_pr[2] = vit3->property().old_pr;
                pr[3] = vit4->property().pr;   old_pr[3] = vit4->property().old_pr;

                pr_vec = _mm_load_ps(&pr[0]);
                old_pr_vec = _mm_load_ps(&old_pr[0]);

                d_vec = _mm_sub_ps(pr_vec, old_pr_vec); // Subtract pr vectors and store in d
                d_vec = _mm_mul_ps(d_vec, d_vec); // Square the values in d_vec
                d_vec = _mm_hadd_ps(_mm_hadd_ps(d_vec, zero_vec), zero_vec); // Sum all elements in d_vec
				
                _mm_store_ss(&result[0], d_vec);

                e_vec_sum += result[0];
            }


            /*for (unsigned vid=start;vid<end;vid++)
            {
                vit = g.find_vertex(vid);
                float d = vit->property().pr - vit->property().old_pr;
                e_vec[tid] += d * d;
            }*/

            //cout<<"pr: "<<vit->property().pr<<"\n";
            //cout<<"sizeof(pr): "<<sizeof(vit->property().pr)<<"\n";

            #pragma omp barrier
            if (tid==0)
            {
                float tot=0;
                for (unsigned i=0;i<threadnum;i++)
                {
                	//printf("inside\n");
                    //tot += e_vec[i];
                    tot += e_vec_sum;
                }
                //printf("outside\n");
                float err = sqrt(tot);
                if (err < quad || (++itercnt) > maxiter)
                {
                    stop = true;
                    //cout<<"== end error: "<<err<<endl;
                }
            }
            #pragma omp barrier
        }

        
        _mm_free(pr);
        _mm_free(old_pr);
        _mm_free(result);
        


#ifdef SIM
        SIM_END(enditer==0);
#endif  
        perf.stop(tid, perf_group);
    }
}
void output(graph_t& g) 
{
    cout<<"Page Rank Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": "<<setprecision(4)
            <<vit->property().pr<<"\n";
    }
}


int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: Degree Centrality\n";

    argument_parser arg;
    gBenchPerf_event perf;
    arg_init(arg);
    if (arg.parse(argc,argv,perf,false)==false)
    {
        arg.help();
        return -1;
    }
    string path, separator;
    arg.get_value("dataset",path);
    arg.get_value("separator",separator);

    size_t threadnum, maxiter;
    arg.get_value("threadnum",threadnum);
    arg.get_value("maxiter",maxiter);
#ifdef SIM
    arg.get_value("beginiter",beginiter);
    arg.get_value("enditer",enditer);
#endif

    double damp, quad;
    arg.get_value("damp",damp);
    arg.get_value("quad",quad);

    double t1, t2;
    graph_t graph;
    cout<<"loading data... \n";
    t1 = timer::get_usec();
    string vfile = path + "/vertex.csv";
    string efile = path + "/edge.csv";

#ifndef EDGES_ONLY
    if (graph.load_csv_vertices(vfile, true, separator, 0) == -1)
        return -1;
    if (graph.load_csv_edges(efile, true, separator, 0, 1) == -1) 
        return -1;
#else
    if (graph.load_csv_edges(efile, true, separator, 0, 1) == -1)
        return -1;
#endif
   
    size_t vertex_num = graph.num_vertices();
    size_t edge_num = graph.num_edges();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    cout<<"threadnum: "<<threadnum<<endl;
    cout<<"damping factor: "<<damp<<endl;
    cout<<"quadratic error: "<<quad<<endl;
    cout<<"\ncomputing Page Rank ...\n";

    gBenchPerf_multi perf_multi(threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() / (double)DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    threadnum = 1; //changed
    for (unsigned i=0;i<run_num;i++)
    {
        init_pagerank(graph, damp, threadnum);

        // Degree Centrality
        t1 = timer::get_usec();
        
        parallel_pagerank(graph, threadnum, damp, quad, maxiter, perf_multi, i);

        t2 = timer::get_usec();
        elapse_time += t2-t1;
    }


    cout<<"Page Rank finish \n";
    cout<<"== iteration #: "<<itercnt<<endl;
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<elapse_time/run_num<<" sec\n";
    if (threadnum == 1)
        perf.print();
    else
        perf_multi.print();

#endif

#ifdef ENABLE_OUTPUT
    cout<<endl;
    output(graph);
#endif
    cout<<"==================================================================\n";
    return 0;
}  // end main

