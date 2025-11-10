// Format date to readable string
export function formatDate(dateString) {
  const date = new Date(dateString);
  return date.toLocaleString('en-IN', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

// Format date for charts (shorter)
export function formatChartDate(dateString) {
  const date = new Date(dateString);
  return date.toLocaleString('en-IN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit'
  });
}

// Round number to decimals
export function roundNumber(num, decimals = 2) {
  if (num === null || num === undefined) return 'N/A';
  return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals);
}
