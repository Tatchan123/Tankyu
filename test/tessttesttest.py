toba_option = { "epsilon":[1e-6,3e-3,1.5e-3,1.2e-3],
                "complement":True,
                "rmw_layer":[2,3,4],
                "delete_n":[0,10,10,7] }

epsilon, complement, rmw_layer = [toba_option[key] for key in ["epsilon", "complement", "rmw_layer"]]
print(epsilon, complement, rmw_layer)