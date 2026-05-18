# comp_left, comp_left!, comp_right, comp_right!
# lcomp_left, lcomp_left!, lcomp_right, lcomp_right!
# comp_res_left, comp_res_left!, comp_res_right, comp_res_right!
# comp_res_prod!
# jac_comp_left_structure, jac_comp_left_structure!, jac_comp_right_structure, jac_comp_right_structure!
# jac_comp_left_coord, jac_comp_left_coord!, jac_comp_right_coord, jac_comp_right_coord!
# comp_residual, comp_residual_product, comp_residual_sum

@testset "MPCCModel Tests" begin
    @testset "VarVar API" for T in [Float64, Float32]
        f(x) = (x[1] - 1)^2 + (x[2] - 1)^2
        ∇f(x) = T[2 * (x[1] - 1); 2 * (x[2] - 1)]
        H(x) = T[2.0 0; 0 2.0]
        c(x) = T[]
        J(x) = T[]
        H(x, y) = H(x)
        cc1(x) = x[1]
        lcc1(x) = T[0.0]
        cc2(x) = x[2]
        lcc2(x) = T[0.0]
        Jcc1(x) = T[1.,0.]
        Jcc2(x) = T[0.,1.]
        comp_res(x) = T[min(x[1],x[2])]
        comp_res_prod(x) = T[x[1]*x[2]]
        comp_res_sum(x) = x[1]*x[2]

        mpcc = SimpleMPCCMode(T)
        n = get_nvar(mpcc)
        m = get_ncon(mpcc)
        ncc = get_ncc(mpcc)
        @test n == 2
        @test m == 0
        @test ncc == 1

        x = randn(T, n)
        y = randn(T, m)
        v = randn(T, n)
        w = randn(T, m)
        Jv = zeros(T, m)
        Jtw = zeros(T, n)
        Hv = zeros(T, n)
        Hvals = zeros(T, get_nnzh(mpcc))

        # NLP api
        @test obj(mpcc, x) ≈ f(x)
        @test grad(mpcc, x) ≈ ∇f(x)
        @test hess(mpcc, x) ≈ H(x)
        @test hprod(mpcc, x, v) ≈ H(x) * v
        @test cons(mpcc, x) ≈ c(x)
        @test jac(mpcc, x) ≈ J(x)
        @test jprod(mpcc, x, v) ≈ J(x) * v
        @test jtprod(mpcc, x, w) ≈ J(x)' * w
        @test hess(mpcc, x, y) ≈ H(x, y)
        @test hprod(mpcc, x, y, v) ≈ H(x, y) * v
        # MPCC api
        @test comp_res_left(mpcc,x) ≈ comp_left(mpcc,x)
        @test comp_res_left(mpcc,x) ≈ cc1(x)
        @test comp_res_right(mpcc,x) ≈ comp_right(mpcc,x)
        @test comp_res_right(mpcc,x) ≈ cc2(x)
        @test jac_comp_left(mpcc,x) ≈ Jcc1(x)
        @test jac_comp_right(mpcc,x) ≈ Jcc2(x)
        @test comp_residual(mpcc,x) ≈ comp_res(x)
        @test comp_residual_prod(mpcc,x) ≈ comp_res_prod(x)
        @test comp_residual_sum(mpcc,x) ≈ comp_res_sum(x)
    end
end
