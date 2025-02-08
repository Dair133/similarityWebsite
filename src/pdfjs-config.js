import { pdfjs } from 'react-pdf';

// Using HTTPS for the worker
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js`;

export default pdfjs;