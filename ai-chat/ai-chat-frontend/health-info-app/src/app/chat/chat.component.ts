import {Component, OnInit} from '@angular/core';
import {ApiService} from "../api.service";  // Import ApiService

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements OnInit {
  question: string = '';
  messages: { text: string, isUser: boolean }[] = [];

  constructor(private apiService: ApiService) {}  // Inject ApiService

  ngOnInit(){ }

  askQuestion(): void {
    if (!this.question.trim()) return;

    // Add user question to messages
    this.messages.push({ text: this.question, isUser: true });

    // Use ApiService to send the question
    this.apiService.askQuestion(this.question).subscribe({
      next: (response) => {
        console.log("Received response:", response);
        // Add AI response to messages
        this.messages.push({ text: response.answer, isUser: false });
        this.question = '';  // Reset the question input
      },
      error: (error) => {
        console.error('Error asking question:', error);
        this.messages.push({ text: 'Failed to get a response.', isUser: false });
      }
    });
  }
}
