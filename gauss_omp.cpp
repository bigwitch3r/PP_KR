#include <iostream>
#include <omp.h>

using namespace std;
bool stop_omp = false;

// Вывод системы уравнений
void sysout_omp(double** a, double* y, int n)
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

double* gauss_omp(double** a, double* y, int n)
{
    double* x, max;
    int k, index;
    const double eps = 0.00001;  // точность

    omp_set_num_threads(8);

    x = new double[n];
    k = 0;

    while (k < n && stop_omp == false)
    {
        // Поиск строки с максимальным a[i][k]
        max = abs(a[k][k]);
        index = k;

#pragma omp parallel for
        for (int i = k + 1; i < n; i++)
        {
            if (abs(a[i][k]) > max)
            {
                max = abs(a[i][k]);
                index = i;
                
            }
        }

        // Перестановка строк
        if (max < eps)
        {
            // нет ненулевых диагональных элементов
            cout << "Решение получить невозможно из-за нулевого столбца ";
            cout << index << " матрицы A" << endl;
            stop_omp = true;
        }
        else
        {
#pragma omp parallel for
            for (int j = 0; j < n; j++)
            {
                double temp = a[k][j];
                a[k][j] = a[index][j];
                a[index][j] = temp;
            }

            double temp = y[k];
            y[k] = y[index];
            y[index] = temp;

            // Нормализация уравнений
#pragma omp parallel for
            for (int i = k; i < n; i++)
            {
                double temp = a[i][k];
                if (abs(temp) < eps) continue; // для нулевого коэффициента пропустить
                for (int j = 0; j < n; j++)
                    a[i][j] = a[i][j] / temp;
                y[i] = y[i] / temp;
                if (i == k)  continue; // уравнение не вычитать само из себя
                for (int j = 0; j < n; j++)
                    a[i][j] = a[i][j] - a[k][j];
                y[i] = y[i] - y[k];
            }
            k++;
        }

    }

    // обратная подстановка
#pragma omp parallel for
    for (k = n - 1; k >= 0; k--)
    {
        x[k] = y[k];
        for (int i = 0; i < k; i++)
            y[i] = y[i] - a[i][k] * x[k];
    }

    return x;
}

int gauss_omp_main()
{

    double** a, * y, * x;
    int n;

    system("chcp 1251");
    system("cls");

    cout << "Введите количество уравнений: ";
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

    // sysout_omp(a, y, n);

    double start_time = omp_get_wtime();

    x = gauss_omp(a, y, n);

    double end_time = omp_get_wtime();
    
    if (stop_omp == false)
    {
        for (int i = 0; i < n; i++)
        {
            cout << "x[" << i << "]=" << x[i] << endl;
        }
    }

    cout << "Time passed: " << end_time - start_time << endl;

    cin.get(); cin.get();
    return 0;
}