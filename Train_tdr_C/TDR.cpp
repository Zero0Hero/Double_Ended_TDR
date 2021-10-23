#include "TDR.h"

#define N_Node 12

Precision NL_Con(Precision J, Precision X_lt);
Precision TDR(Precision U_x, Precision x_lt[N_Node], Precision x_t[N_Node], Precision M[N_Node], Precision W[N_Node], char mode);

Precision TDR(Precision U_x, Precision x_lt[N_Node], Precision x_t[N_Node], Precision M[N_Node] ,Precision W[N_Node], char mode)
{
	if (mode == 0) //predicting mode
	{
		Precision U_z_p=0;
		for (int i = 0; i < N_Node; i++)
		{
			Precision J = M[i] * U_x;
			x_t[i] = NL_Con( J, x_lt[i]);
			U_z_p += W[i] * x_t[i];
		}
		return U_z_p;
	}
	else if (mode == 1)//training mode
	{
		for (int i = 0; i < N_Node; i++)
		{
			Precision J = M[i] * U_x;
			x_t[i] = NL_Con(J, x_lt[i]);
		}
		return 0;
	}
	else //if (mode == 2) //training output mode
	{
		Precision U_z_p = 0;
		for (int i = 0; i < N_Node; i++)
		{
			U_z_p += W[i] * x_t[i];
		}
		return U_z_p;
	}
}





//Nonlinear conversion
#define n 2.8
#define p 4
#define r 2
#define h 0.03
Precision NL_Con(Precision J, Precision X_lt)
{
	Precision temp = X_lt + r * J;
	Precision k = n * (temp) / (1 + pow(temp, p));
	Precision k1 = k - n * X_lt;
	Precision k2 = k - n * (X_lt + k1 * h / 2);
	Precision k3 = k - n * (X_lt + k2 * h);
	Precision k4 = k - n * (X_lt + k3 * h);
	return (X_lt + (k1 + k2 * 2 + k3 * 2 + k4) * h / 6);
}