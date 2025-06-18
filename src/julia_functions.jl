# import Pkg; Pkg.add("EquationsOfStateOfSolids")
using ITensors, ITensorMPS
using PastaQ
using EquationsOfStateOfSolids
using EquationsOfStateOfSolids.Fitting

function fit_Murnaghan(volumes,energies,v0,b0,b0_p,e0)
  res_array= Vector{Any}()
  results = nonlinfit(EnergyEquation(Murnaghan(v0, b0, b0_p , e0)), volumes, energies)
  # println("result: ",results[1])
  push!(res_array,results.v0)
  push!(res_array,results.b0)
  push!(res_array,results.b′0)
  push!(res_array,results.e0)
  return res_array
end

function my_dmrg(N,ham,nsweeps;maxdim=[1000],cutoff=[1e-32])
  # sites = siteinds("Qubit",N,conserve_qns=true)#,conserve_number=true)
  sites = [(n<=N/2) ? siteind("Qubit", n; conserve_number=true, qnname_number="SpinUp") : siteind("Qubit", n; conserve_number=true, qnname_number="SpinDn") for n in 1:N]
  # sites= siteinds(n->(n<=N/2) ? "S=1/2" : "S=1/2",N)
#   println(typeof(ham),length(ham),ham)
  os = OpSum()
  for ham_term_counter=1:length(ham)
    os += ham[ham_term_counter]
  end
  # println("os:   ",os)
  # print("flux H: ",flux(os))
  H = MPO(os, sites)
  
#   nsweeps = 5 # number of sweeps is 5
#   maxdim = [10,20,100,100,200] # gradually increase states kept
#   cutoff = [1E-10] # desired truncation error

  println("N: ",N/4,"  ",N/2,"  ",N/2 + N/4)
  state1 = [isodd(n) ? "Up" : "Dn" for n=1:N] 
  state2 = [ (n<=N/4) || ( (n>N/2) && (n<=(N/2 + N/4)) ) ? "0" : "1" for n=1:N]
  state3 = [ (n<=N/4) || ( (n>N/2) && (n<=(N/2 + N/4)) ) ? "1" : "0" for n=1:N]
  state4 = [isodd(n) ? "Dn" : "Up" for n=1:N] 
  state5 = [ (n<=N/4) || ((n>(N/2 + N/4))) ? "0" : "1" for n=1:N]
  state6 = [ (n<=N/4) || ((n>(N/2 + N/4))) ? "1" : "0" for n=1:N]
  state7 = [ (n % 4)==0 || (n % 4)==1 ? "1" : "0" for n=1:N]
  state8 = [ (n % 4)==0 || (n % 4)==1 ? "0" : "1" for n=1:N] 
  state9 = [ ((n-1) % 4)==0 || ((n-1) % 4)==1 ? "1" : "0" for n=1:N]
  state10 = [ ((n-1) % 4)==0 || ((n-1) % 4)==1 ? "0" : "1" for n=1:N]  


#   println("state: ",state)
  # println("flux: ",flux(state))
  psi1 = MPS(sites,state1)
  psi2 = MPS(sites,state2)
  psi3 = MPS(sites,state3)
  psi4 = MPS(sites,state4)
  psi5 = MPS(sites,state5)
  psi6 = MPS(sites,state6)
  psi7 = MPS(sites,state7)
  psi8 = MPS(sites,state8)
  psi9 = MPS(sites,state9)
  psi10 = MPS(sites,state10)
  # samples=getsamples(psi1,10)
  # println("psi1 samples: ",samples)
  # samples=getsamples(psi2,10)
  # println("psi2 samples: ",samples)
  # samples=getsamples(psi3,10)
  # println("psi3 samples: ",samples)
  # samples=getsamples(psi4,10)
  # println("psi4 samples: ",samples)
  # samples=getsamples(psi5,10)
  # println("psi5 samples: ",samples)
  # samples=getsamples(psi7,2)
  # println("psi7 samples: ",samples)
  # samples=getsamples(psi8,2)
  # println("psi8 samples: ",samples)
  # samples=getsamples(psi9,2)
  # println("psi9 samples: ",samples)
  # samples=getsamples(psi10,2)
  # println("psi10 samples: ",samples)

  if N<=20
    psi0 = 1.69817*psi1+0.734837*psi4+1.4893*psi3+0.8131*psi2
  elseif N%8 == 0
    psi0 = 1.69817*psi1+0.734837*psi4+1.4893*psi3+0.8131*psi2 + psi5 + psi6  + psi7 + psi8 + psi9 + psi10
  else
    psi0 = 1.69817*psi1+0.734837*psi4+1.4893*psi3+0.8131*psi2 + psi5 + psi6
  end  
  # psi0 = psi0+psi3
  # psi0 = psi0+psi4
  # psi0 = psi0+psi5
  # psi0 = psi0+psi6
  normalize!(psi0)
  println("norm; ",norm(psi0))

  # psi0 = random_mps(sites,state;linkdims=2)
  println("flux: ",flux(psi0))
  # println("random psi: ",psi0)

  energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)
  samples=getsamples(psi,100)
  # println("psi samples: ",samples)
  println(inner(psi',H,psi))
  #dense_wf = contract(psi)
  #return energy, Array(dense_wf,inds(dense_wf))
  # mps_array = Array(psi[1],inds(psi[1]))
  # println("inds: ",inds(psi[1]))
  # println("",inds(psi[2]))
  # println("",inds(psi[4]))
  mps_array= Vector{Any}()
  for mps_counter in 1:length(psi)
    push!(mps_array,(Array(psi[mps_counter],inds(psi[mps_counter]))))
  end
  return energy,psi
  # return energy, mps_array
  # return
end

function my_dmrg_uhf(N,nelec_a,nelec_b,ham,nsweeps;maxdim=[1000],cutoff=[1e-32])
  # sites = siteinds("Qubit",N,conserve_qns=true)#,conserve_number=true)
  sites = [(n<=N/2) ? siteind("Qubit", n; conserve_number=true, qnname_number="SpinUp") : siteind("Qubit", n; conserve_number=true, qnname_number="SpinDn") for n in 1:N]
  # sites= siteinds(n->(n<=N/2) ? "S=1/2" : "S=1/2",N)
#   println(typeof(ham),length(ham),ham)
  os = OpSum()
  for ham_term_counter=1:length(ham)
    os += ham[ham_term_counter]
  end
  # println("os:   ",os)
  # print("flux H: ",flux(os))
  H = MPO(os, sites)
  
#   nsweeps = 5 # number of sweeps is 5
#   maxdim = [10,20,100,100,200] # gradually increase states kept
#   cutoff = [1E-10] # desired truncation error

  # println("N: ",N/4,"  ",N/2,"  ",N/2 + N/4)
  # state1 = [isodd(n) ? "Up" : "Dn" for n=1:N] 
  state1 = [ (n<=nelec_a) || ( (n>N/2) && (n<=(N/2 + nelec_b)) ) ? "0" : "1" for n=1:N]
  println("state1: ",state1)
  state2 = [ (n>=(N/2 - nelec_a)+1) && (n<=N/2) || ( (n>N - nelec_b)) ? "0" : "1" for n=1:N]
  println("state2: ",state2)
  state3 = [ ((n<=nelec_a-1) || n==N/2) || ( (n>N/2) && (n<=(N/2 + nelec_b)) ) ? "0" : "1" for n=1:N]
  println("state3: ",state3)
  state4 = [ ((n<=nelec_a)) || ( (n>N/2) && (n<=(N/2 + nelec_b-1)) ) || n==N ? "0" : "1" for n=1:N]
  println("state4: ",state4)
  state5 = [ ((n<=nelec_a-1)) || n==N/2 || ( (n>N/2) && (n<=(N/2 + nelec_b-1)) ) || n==N ? "0" : "1" for n=1:N]
  println("state5: ",state5)
  # state3 = [ (n<=N/4) || ( (n>N/2) && (n<=(N/2 + N/4)) ) ? "1" : "0" for n=1:N]
  # state4 = [isodd(n) ? "Dn" : "Up" for n=1:N] 
  # state5 = [ (n<=N/4) || ((n>(N/2 + N/4))) ? "0" : "1" for n=1:N]
  # state6 = [ (n<=N/4) || ((n>(N/2 + N/4))) ? "1" : "0" for n=1:N]
  # state7 = [ (n % 4)==0 || (n % 4)==1 ? "1" : "0" for n=1:N]
  # state8 = [ (n % 4)==0 || (n % 4)==1 ? "0" : "1" for n=1:N] 
  # state9 = [ ((n-1) % 4)==0 || ((n-1) % 4)==1 ? "1" : "0" for n=1:N]
  # state10 = [ ((n-1) % 4)==0 || ((n-1) % 4)==1 ? "0" : "1" for n=1:N]  


#   println("state: ",state)
  # println("flux: ",flux(state))
  psi1 = MPS(sites,state1)
  psi2 = MPS(sites,state2)
  psi3 = MPS(sites,state3)
  # psi4 = MPS(sites,state4)
  # psi5 = MPS(sites,state5)
  # psi6 = MPS(sites,state6)
  # psi7 = MPS(sites,state7)
  # psi8 = MPS(sites,state8)
  # psi9 = MPS(sites,state9)
  # psi10 = MPS(sites,state10)
  # samples=getsamples(psi1,10)
  # println("psi1 samples: ",samples)
  # samples=getsamples(psi2,10)
  # println("psi2 samples: ",samples)
  # samples=getsamples(psi3,10)
  # println("psi3 samples: ",samples)
  # samples=getsamples(psi4,10)
  # println("psi4 samples: ",samples)
  # samples=getsamples(psi5,10)
  # println("psi5 samples: ",samples)
  # samples=getsamples(psi7,2)
  # println("psi7 samples: ",samples)
  # samples=getsamples(psi8,2)
  # println("psi8 samples: ",samples)
  # samples=getsamples(psi9,2)
  # println("psi9 samples: ",samples)
  # samples=getsamples(psi10,2)
  # println("psi10 samples: ",samples)

  # if N<=20
  psi0 = psi1+psi2+psi3
  # elseif N%8 == 0
  #   psi0 = 1.69817*psi1+0.734837*psi4+1.4893*psi3+0.8131*psi2 + psi5 + psi6  + psi7 + psi8 + psi9 + psi10
  # else
  #   psi0 = 1.69817*psi1+0.734837*psi4+1.4893*psi3+0.8131*psi2 + psi5 + psi6
  # end  
  # psi0 = psi0+psi3
  # psi0 = psi0+psi4
  # psi0 = psi0+psi5
  # psi0 = psi0+psi6
  normalize!(psi0)
  println("norm; ",norm(psi0))

  # psi0 = random_mps(sites,state;linkdims=2)
  println("flux: ",flux(psi0))
  # println("random psi: ",psi0)

  energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)
  samples=getsamples(psi,100)
  # println("psi samples: ",samples)
  println(inner(psi',H,psi))
  #dense_wf = contract(psi)
  #return energy, Array(dense_wf,inds(dense_wf))
  # mps_array = Array(psi[1],inds(psi[1]))
  # println("inds: ",inds(psi[1]))
  # println("",inds(psi[2]))
  # println("",inds(psi[4]))
  mps_array= Vector{Any}()
  for mps_counter in 1:length(psi)
    push!(mps_array,(Array(psi[mps_counter],inds(psi[mps_counter]))))
  end
  return energy,psi
  # return energy, mps_array
  # return
end

function make_tau_mps(mps_qn)
  sites = siteinds(mps_qn)
  state0 = ["0" for n=1:length(mps_qn)] 
  psi0 = MPS(sites,state0)
  tau_mps = mps_qn + psi0
  normalize!(tau_mps)
  return tau_mps
end

function cliff_and_meas(mps_qn,gate_names,gate_qubits;maxdim=0,cutoff=0)
  # println(gate_names[1],gate_qubits[1])
  circuit_gates=Vector{Any}()
  for n=1:length(gate_names)
    # println(n,"  ",gate_names[n],"  ",gate_qubits[n])
    push!(circuit_gates,(gate_names[n],gate_qubits[n]))
  end
#   println(circuit_gates)
  mps = dense(mps_qn) ##remove symmetry

  if (maxdim == 0 && cutoff==0)
    final_mps = PastaQ.runcircuit(mps,circuit_gates)
  elseif maxdim != 0 && cutoff==0
    final_mps = PastaQ.runcircuit(mps,circuit_gates,maxdim=maxdim)
  elseif maxdim == 0 && cutoff!=0 
    final_mps = PastaQ.runcircuit(mps,circuit_gates,cutoff=cutoff)
  elseif maxdim != 0 && cutoff!=0 
    final_mps = PastaQ.runcircuit(mps,circuit_gates,cutoff=cutoff,maxdim=maxdim)  
  end  

  normalize!(final_mps)
  sample=getsamples(final_mps,1)
  return sample
end



function get_SV_from_mps(mps)
  SV_array= Vector{Any}()
  SV=contract(mps)
  # println("length mps SV = ", length(SV))
  for mps_counter in 1:2^length(mps)
    push!(SV_array,SV[mps_counter])
  end
  return SV_array
end