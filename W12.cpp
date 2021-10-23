/*
�㷨����
���ն���ļ��㷽���˷����㣬����Ӱ�������ܡ�����Ҫ����Billboard��������ʱ������������Ż��ܼ���������ܡ�
����Ҫ���ܵľ��������㷨��Ϊȫѡ��Ԫ��˹-Լ������

��˹-Լ������ȫѡ��Ԫ������Ĳ������£�

���ȣ����� k �� 0 �� N - 1 �����¼�����

�ӵ� k �С��� k �п�ʼ�����½�������ѡȡ����ֵ����Ԫ�أ�����ס��Ԫ�����ڵ��кź��кţ�
��ͨ���н������н���������������Ԫ��λ���ϡ���һ����Ϊȫѡ��Ԫ��
m(k, k) = 1 / m(k, k)
m(k, j) = m(k, j) * m(k, k)��j = 0, 1, ..., N-1��j != k
m(i, j) = m(i, j) - m(i, k) * m(k, j)��i, j = 0, 1, ..., N-1��i, j != k
m(i, k) = -m(i, k) * m(k, k)��i = 0, 1, ..., N-1��i != k
��󣬸�����ȫѡ��Ԫ����������¼���С��н�������Ϣ���лָ����ָ���ԭ�����£�
��ȫѡ��Ԫ�����У��Ƚ������У��У�����лָ���ԭ�����У��У��������У��У��������ָ���
*/
#include <iostream>
#include "stdio.h"
#include <string.h>
#include "w.h"
#define N 12
#define M 6000
void _init(Precision in[N][N], Precision a[N][N]);
Precision   choose_the_main(Precision a[N][N], Precision b[N], short   k);
void   input_output(Precision a[N][N], Precision c[N][N], Precision q[N][N]);
void   course(Precision a[N][N], Precision c[N][N], Precision q[N][N]);
void inv(Precision a[N][N], Precision out[N][N]);
void W_N_12(Precision X[N][M], Precision W[N], Precision Y[M]);

void W_N_12(Precision X[N][M], Precision W[N], Precision Y[M])//,Precision Y[N][N]
{
	Precision Z[N][N], T[N][N], R[N];//Z[N][N],,X[N][M]
	int j = 0, i = 0, k = 0;

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			Precision sum = 0;
			for (k = 0; k < M; k++)
				sum += X[i][k] * X[j][k];
			Z[i][j] = sum;
			if (i == j)
			{
				Z[i][j] += 0.000001;
			}
		}
	}

	//���Z��ΪT
	//inv(Z, T);
	Precision c[N][N], q[N][N];//
	input_output(Z, c, q);
	course(Z, c, q);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			T[i][j] = c[i][j];
		}
	//���Y��Xt֮��R
	for (j = 0; j < N; j++)
	{
		Precision sum = 0;
		for (k = 0; k < M; k++)
			sum += Y[k] * X[j][k];
		R[j] = sum;
	}

	//W=R*T

	for (j = 0; j < N; j++)
	{
		Precision sum = 0;
		for (k = 0; k < N; k++)
			sum += R[k] * T[j][k];
		W[j] = sum;
	}
}
void inv(Precision in[N][N], Precision out[N][N])
{

	Precision  a[N][N], c[N][N], q[N][N];//
	_init(in, a);
	input_output(a, c, q);
	course(a, c, q);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			out[i][j] = c[i][j];
		}
}

Precision   choose_the_main(Precision a[N][N], Precision b[N], short   k)
{
	Precision   d, t;
	int   l, i, j;
	d = a[k - 1][k - 1];
	l = k - 1;
	for (i = 0; i < N; i++)//i=k
	{
		if (i > k - 1)
			if (fabs(a[i][k - 1] > fabs(d)))
			{
				d = a[i][k - 1];
				l = i;
			}
	}

	if (l != k - 1)
	{
		for (j = 0; j < N; j++)//j = k - 1
		{
			if (j > k - 2)
			{
				t = a[l][j];
				a[l][j] = a[k - 1][j];
				a[k - 1][j] = t;
			}
		}
		t = b[l]; b[l] = b[k - 1]; b[k - 1] = t;
	}
	return   (d);
}

void   input_output(Precision a[N][N], Precision c[N][N], Precision q[N][N])
{

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			if (i == j)   c[i][j] = 1;
			else   c[i][j] = 0;
			q[i][j] = a[i][j];
		}
}

void  course(Precision a[N][N], Precision c[N][N], Precision q[N][N])
{
	Precision   sum = 0, h; Precision b[N] = { 0 };
	short   i, j, k, p, flag = 0;
	for (p = 0; p < N; p++)
	{
		for (j = 0; j < N; j++)
			b[j] = c[j][p];
		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++)
			{
				a[i][j] = q[i][j];
			}


		for (k = 1; k < N; k++)
		{

			h = choose_the_main(a, b, k);
			if (h == 0)
			{
				flag = 1;
				break;
			}
			else
			{
				for (i = k; i < N; i++)
					a[i][k - 1] = a[i][k - 1] / a[k - 1][k - 1];
				for (i = k; i < N; i++)
					for (j = k; j < N; j++)
						a[i][j] = a[i][j] - a[i][k - 1] * a[k - 1][j];
				for (i = k; i < N; i++)
					b[i] = b[i] - (a[i][k - 1] * b[k - 1]);
			}
		}
		if (flag == 1)   break;


		if (h != 0)
		{
			b[N - 1] = b[N - 1] / a[N - 1][N - 1];
			for (i = N - 2; i >= 0; i--)
			{
				for (j = i + 1; j < N; j++)
					sum = sum + a[i][j] * b[j];
				b[i] = (b[i] - sum) / a[i][i];
				sum = 0;
			}
			for (j = 0; j < N; j++)
				c[j][p] = b[j];

		}
	}
}

void _init(Precision in[N][N], Precision a[N][N])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i][j] = in[i][j];
		}
}
