import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { formatChartDate } from '../utils/helpers';
import { getAQIStyle } from '../utils/constants';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function ForecastChart({ predictions }) {
  // Get colors based on AQI categories
  const backgroundColors = predictions.map(pred => {
    const style = getAQIStyle(pred.predicted_aqi);
    return style.color + '80'; // Add transparency
  });

  const borderColors = predictions.map(pred => {
    const style = getAQIStyle(pred.predicted_aqi);
    return style.color;
  });

  const chartData = {
    labels: predictions.map(p => formatChartDate(p.target_timestamp)),
    datasets: [
      {
        label: 'Predicted AQI',
        data: predictions.map(p => p.predicted_aqi),
        backgroundColor: backgroundColors,
        borderColor: borderColors,
        borderWidth: 2,
        borderRadius: 4
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        callbacks: {
          label: function(context) {
            const pred = predictions[context.dataIndex];
            return [
              `AQI: ${pred.predicted_aqi.toFixed(1)}`,
              `Category: ${pred.predicted_category}`,
              `Model: ${pred.model_type}`
            ];
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45
        }
      },
      y: {
        beginAtZero: true,
        max: 500,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        title: {
          display: true,
          text: 'Predicted AQI'
        }
      }
    }
  };

  return (
    <div className="chart-wrapper">
      <Bar data={chartData} options={options} />
      <div className="forecast-legend">
        <p>
          🤖 Predictions generated using <strong>{predictions[0]?.model_type || 'ML model'}</strong>
        </p>
      </div>
    </div>
  );
}

export default ForecastChart;
