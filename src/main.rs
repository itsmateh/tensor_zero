use tensorlab::Tensor;
use std::time::Instant;
fn main(){
    println!("TensorLab, the tensor operations project!");
    // println!("matmul benchmark:");
    // let size = 4096; 
    // println!("generating {}x{} matrices", size, size);
    
    // let data_a = vec![1.0; size * size];
    // let data_b = vec![1.0; size * size];
    // let shape = vec![size, size];

    // let a = Tensor::new(&data_a, &shape).unwrap();
    // let b = Tensor::new(&data_b, &shape).unwrap();

    // let start = Instant::now();
    // let _ = a.matmul_naive(&b).unwrap();
    // let t_naive = start.elapsed().as_secs_f64();
    // println!("1. naive (i-j-k):        {:.4} s", t_naive);

    // let start = Instant::now();
    // let _ = a.matmul_mem_opt(&b).unwrap();
    // let t_trans = start.elapsed().as_secs_f64();
    // println!("2. transpose (Alloc, i-j-k):    {:.4} s", t_trans);

    // let start = Instant::now();
    // let _ = a.matmul_mem_opt_ikj(&b).unwrap();
    // let t_ikj = start.elapsed().as_secs_f64();
    // println!("3. tranpose (Alloc, i-k-j):    {:.4} s", t_ikj);

    // let start = Instant::now();
    // let _ = a.matmul(&b).unwrap();
    // let t_par = start.elapsed().as_secs_f64();
    // println!("4. parallel with Rayon (i-j-k):       {:.4} s", t_par);

    // let start = Instant::now();
    // let _ = a.matmul_parallel_ikj(&b).unwrap();
    // let t_par_ikj = start.elapsed().as_secs_f64();
    // println!("5. parallel with Rayon (i-k-j):       {:.4} s", t_par_ikj);

    // println!("\n--- Final Analysis ---");
    
    // // Cuánto mejoró IKJ respecto a Naive (Optimización pura de algoritmo)
    // println!("Algorithmic Speedup (Naive -> IKJ):    {:.2}x faster", t_naive / t_ikj);
    
    // // Cuánto mejoró Paralelo respecto al MEJOR Serial (IKJ)
    // println!("Parallel Scaling (IKJ -> Parallel):    {:.2}x faster", t_ikj / t_par);
    
    // // Ganancia Total
    // println!("Total Speedup (Naive -> Parallel):     {:.2}x faster", t_naive / t_par);
    
    // println!("\n--- Final Truth ---");
    // println!("Rayon vs IKJ Serial:   {:.2}x faster", t_ikj / t_par_ikj);
}
