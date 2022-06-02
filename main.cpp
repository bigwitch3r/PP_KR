#include <iostream>
#include <omp.h>
#include "gauss_omp.h"
#include "gauss_mpi.h"

using namespace std;
bool stop = false;

// ����� ������� ���������
void sysout_serial(double** a, double* y, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << a[i][j] << "*x" << j;
            if (j < n - 1)
                cout << " + ";
        }
        cout << " = " << y[i] << endl;
    }
    return;
}

double* gauss_serial(double** a, double* y, int n)
{
    double* x, max;
    int k, index;
    const double eps = 0.00001;  // ��������

    x = new double[n];
    k = 0;

    while (k < n)
    {
        // ����� ������ � ������������ a[i][k]
        max = abs(a[k][k]);
        index = k;

        for (int i = k + 1; i < n; i++)
        {
            if (abs(a[i][k]) > max)
            {
                max = abs(a[i][k]);
                index = i;
            }
        }

        // ������������ �����
        if (max < eps)
        {
            // ��� ��������� ������������ ���������
            cout << "������� �������� ���������� ��-�� �������� ������� ";
            cout << index << " ������� A" << endl;
            stop = true;
            return 0;
        }

        for (int j = 0; j < n; j++)
        {
            double temp = a[k][j];
            a[k][j] = a[index][j];
            a[index][j] = temp;
        }

        double temp = y[k];
        y[k] = y[index];
        y[index] = temp;

        // ������������ ���������
        for (int i = k; i < n; i++)
        {
            double temp = a[i][k];
            if (abs(temp) < eps) continue; // ��� �������� ������������ ����������
            for (int j = 0; j < n; j++)
                a[i][j] = a[i][j] / temp;
            y[i] = y[i] / temp;
            if (i == k)  continue; // ��������� �� �������� ���� �� ����
            for (int j = 0; j < n; j++)
                a[i][j] = a[i][j] - a[k][j];
            y[i] = y[i] - y[k];
        }
        k++;
    }

    // �������� �����������
    for (k = n - 1; k >= 0; k--)
    {
        x[k] = y[k];
        for (int i = 0; i < k; i++)
            y[i] = y[i] - a[i][k] * x[k];
    }

    return x;
}

void serial()
{
    double** a, * y, * x;
    int n;

    system("chcp 1251");
    system("cls");

    cout << "������� ���������� ���������: ";
    cin >> n;

    a = new double* [n];

    for (int i = 0; i < n; i++)
    {
        a[i] = new double[n];
    }

    y = new double[n];
    double range = 1.0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            a[i][j] = range * (1.0 - 2.0 * (double)rand() / RAND_MAX);
        y[i] = range * (1.0 - 2.0 * (double)rand() / RAND_MAX);
    }

    /*for (int i = 0; i < n; i++)
    {
        a[i] = new double[n];
        for (int j = 0; j < n; j++)
        {
            cout << "a[" << i << "][" << j << "]= ";
            cin >> a[i][j];
        }
    }

    for (int i = 0; i < n; i++)
    {
        cout << "y[" << i << "]= ";
        cin >> y[i];
    }*/

    // sysout_serial(a, y, n);

    double start_time = omp_get_wtime();

    x = gauss_serial(a, y, n);

    double end_time = omp_get_wtime();

    if (stop == false)
    {
        for (int i = 0; i < n; i++)
        {
            cout << "x[" << i << "]=" << x[i] << endl;
        }
    }

    cout << "Time passed: " << end_time - start_time << endl;
}


int main(int* argc, char** argv)
{
    gauss_omp_main();
    // serial();
    // gauss_mpi(argc, argv);

    cin.get(); cin.get();
    return 0;
}