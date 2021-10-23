#include <iostream>
#include "stdio.h"
#include "w.h"
#include "TDR.h"
#include "main.h"

#define N_Node 12
#define N_TDR 4
#define N_Train 6000
#define N_Discard 100
#define N_Test 10000

int main()
{
	Precision U1_x[N_TDR][N_Train + N_Discard] = { 0 };//1-6100
	Precision U1_z[N_Train] = { 0 }; //101-6100

	Precision Mask[N_Node] = { -0.194067297054429,0.362223362359113,0.928811548786572,-0.065866510724707,
			-0.787234831401828,0.666497251492163,-0.898636738808856,-0.100252622669462,
			0.892916665924410,-0.989651135684455,-0.436819777210518,-0.305799057021391 };

	Precision x_t[N_TDR* N_Node][N_Train] = { 0 };//state of TDRs for training 48*6000

	Precision W[N_TDR-1][N_Node] = { 0 }; //output weight of each TDR
	Precision W_t[N_TDR * N_Node];//ouput weight of final output layer
	//read training-data
	{
		FILE* fp = 0;
		fopen_s(&fp, "F:\\VS\\data\\U1_x.txt", "r");

		for (int j = 0; j < N_Train + N_Discard; j++)
		{
			fscanf_s(fp, "%lf ", &U1_x[0][j]);
		}
		fclose(fp);

		FILE* fq = 0;
		fopen_s(&fq, "F:\\VS\\data\\U1_z.txt", "r");

		for (int j = 0; j < N_Train; j++)
		{
			fscanf_s(fq, "%lf ", &U1_z[j]);
		}
		fclose(fq);
	}

	Precision temp_x[N_Train + N_Discard][N_Node] = {0};//x state to next
	Precision temp_lx[N_TDR][N_Node] = { -0.194067297054429,0.362223362359113,0.928811548786572,-0.065866510724707,
			-0.787234831401828,0.666497251492163,-0.898636738808856,-0.100252622669462,
			0.892916665924410,-0.989651135684455,-0.436819777210518,-0.305799057021391,
			-0.194067297054429,0.362223362359113,0.928811548786572,-0.065866510724707,
			-0.787234831401828,0.666497251492163,-0.898636738808856,-0.100252622669462,
			0.892916665924410,-0.989651135684455,-0.436819777210518,-0.305799057021391, 
			-0.194067297054429,0.362223362359113,0.928811548786572,-0.065866510724707,
			-0.787234831401828,0.666497251492163,-0.898636738808856,-0.100252622669462,
			0.892916665924410,-0.989651135684455,-0.436819777210518,-0.305799057021391, 
			-0.194067297054429,0.362223362359113,0.928811548786572,-0.065866510724707,
			-0.787234831401828,0.666497251492163,-0.898636738808856,-0.100252622669462,
			0.892916665924410,-0.989651135684455,-0.436819777210518,-0.305799057021391 };

	for (int i = 0; i < N_TDR; i++)//TDR i
	{
		for (int j = 0; j < N_Train+N_Discard; j++)//Train j
		{
			if (j < N_Discard)
			{
				TDR(U1_x[i][j], temp_lx[i], temp_x[j], Mask, NULL, 1);
				for (int k = 0; k < N_Node; k++)
				{
					temp_lx[i][k] = temp_x[j][k];
				}
			}
			else
			{
				TDR(U1_x[i][j], temp_lx[i], temp_x[j], Mask, NULL, 1);
				for (int k = 0; k < N_Node; k++)
				{
					temp_lx[i][k] = temp_x[j][k];
					x_t[i * N_Node + k][j - N_Discard] = temp_x[j][k];
				}
			}	
		}
		
		if (i < N_TDR - 1)
		{
			W_N_12(&(x_t[i * N_Node]), W[i], U1_z);

			for (int j = 0; j < N_Train + N_Discard; j++)
			{
				U1_x[i + 1][j] = TDR(NULL, NULL, temp_x[j], NULL, W[i], 2);
			}

		}

	}
	W_N_48(x_t, W_t, U1_z);
	getchar();
	return 0;
}


