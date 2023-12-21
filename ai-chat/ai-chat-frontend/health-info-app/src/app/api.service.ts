import { Injectable } from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';
import { Observable } from 'rxjs'

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly apiUrl = 'http://127.0.0.1:5002';
  private httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json',
      'Accept': 'application/json'  // This line is sometimes necessary
    })
  };

  constructor(private http: HttpClient) {}

  askQuestion(question: string): Observable<any> {
    const endpoint = `${this.apiUrl}/ask`;
    const body = { question };
    return this.http.post(endpoint, body, this.httpOptions);
  }

}
