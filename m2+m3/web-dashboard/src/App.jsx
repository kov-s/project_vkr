import React, { useEffect, useState } from 'react';
import Papa from 'papaparse';
import { Input } from '@/components/ui/input';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import { Card, CardContent } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { ScrollArea } from '@/components/ui/scroll-area';

export default function App() {
  const [data, setData] = useState([]);
  const [filterText, setFilterText] = useState('');
  const [selectedTopic, setSelectedTopic] = useState('all');
  const [topics, setTopics] = useState([]);

  useEffect(() => {
    fetch('/data/messages_with_topics.csv')
      .then(res => res.text())
      .then(csv => {
        Papa.parse(csv, {
          header: true,
          complete: (results) => {
            const cleanData = results.data.filter(row => row.topic !== '');
            setData(cleanData);
            const uniqueTopics = [...new Set(cleanData.map(row => row.topic))].sort((a, b) => a - b);
            setTopics(uniqueTopics);
          }
        });
      });
  }, []);

  const filteredData = data
    .filter(row =>
      (selectedTopic === 'all' || row.topic === selectedTopic) &&
      (row.text?.toLowerCase().includes(filterText.toLowerCase()))
    )
    .sort((a, b) => parseFloat(b.topic_probability) - parseFloat(a.topic_probability));

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">📊 Тематическое моделирование сообщений</h1>

      <div className="flex flex-wrap gap-4 mb-6">
        <Input
          placeholder="Поиск по тексту..."
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          className="w-full sm:w-64"
        />

        <Select value={selectedTopic} onValueChange={setSelectedTopic}>
          <SelectTrigger className="w-64">
            <SelectValue placeholder="Фильтр по теме" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Все темы</SelectItem>
            {topics.map(topic => (
              <SelectItem key={topic} value={topic}>
                Тема {topic}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <ScrollArea className="border rounded-2xl shadow max-h-[600px]">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>ID</TableHead>
              <TableHead>Оригинальный текст</TableHead>
              <TableHead>Предобработка</TableHead>
              <TableHead>Тема</TableHead>
              <TableHead>Вероятность</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredData.map((row, i) => (
              <TableRow key={i}>
                <TableCell>{row.message_id}</TableCell>
                <TableCell>{row.text}</TableCell>
                <TableCell>{row.processed_text}</TableCell>
                <TableCell>{row.topic}</TableCell>
                <TableCell>{parseFloat(row.topic_probability).toFixed(3)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </ScrollArea>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-10">
        <Card>
          <CardContent className="p-0">
            <iframe src="/intertopic_distance_map.html" className="w-full h-[500px]" title="Intertopic Map" />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-0">
            <iframe src="/topic_word_barchart.html" className="w-full h-[500px]" title="Topic Barchart" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
