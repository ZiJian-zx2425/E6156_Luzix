// src/app/services/auth/auth.service.ts
import { Injectable } from '@angular/core';
import {HttpClient, HttpErrorResponse} from '@angular/common/http';
import {catchError, Observable, throwError} from "rxjs";
import { map, tap } from 'rxjs/operators';
import {JwtPayload} from "../../jwt-payload.model";
import { Router } from '@angular/router';

@Injectable({
    providedIn: 'root'
})
export class AuthService {
    private readonly isLoggedInUrl = 'https://crxrcf7ds8.execute-api.us-east-1.amazonaws.com/test/api/is_logged_in';

    constructor(private http: HttpClient, private router: Router) {}

    private parseJwt(token: string): JwtPayload | null {
        try {
            const base64Url = token.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            const jsonPayload = decodeURIComponent(atob(base64).split('').map(c => {
                return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
            }).join(''));

            const decoded = JSON.parse(jsonPayload) as JwtPayload;
            console.log("Decoded JWT:", decoded); // For debugging
            return decoded;
        } catch (error) {
            console.error("Error parsing JWT", error);
            return null;
        }
    }

    isLoggedIn(): boolean {
        const token = localStorage.getItem('authToken');
        if (!token) {
            return false;
        }

        const decodedToken = this.parseJwt(token) as any;
        if (!decodedToken) {
            return false;
        }

        const expirationDate = new Date(decodedToken.exp * 1000);
        if (expirationDate < new Date()) {
            localStorage.removeItem('authToken');
            return false;
        }

        return true;
    }

    storeToken(token: string): void {
        localStorage.setItem('authToken', token); // Storing the token in local storage
    }

    getUserRole(): string {
        const token = localStorage.getItem('authToken');
        if (token) {
            const decodedToken = this.parseJwt(token);
            if (decodedToken) {
                return decodedToken.role;
            }
        }
        return 'not logged in';
    }

    getUserGoogleId(): string {
        const token = localStorage.getItem('authToken');
        if (token) {
            const decodedToken = this.parseJwt(token);
            if (decodedToken) {
                // Assuming the Google ID is stored in a field named 'google_id' in the token
                return decodedToken.sub || 'not available';
            }
        }
        return 'not available';
    }

    logout(): void {
        // Clear local storage or any other stored user data
        localStorage.removeItem('authToken');
        localStorage.removeItem('userRole');

        // Optionally call the backend logout endpoint
        this.http.get('https://crxrcf7ds8.execute-api.us-east-1.amazonaws.com/test/logout').subscribe(() => {
            // Redirect to login or home page after successful logout
            this.router.navigate(['https://crxrcf7ds8.execute-api.us-east-1.amazonaws.com/test/login']); // Replace '/login' with your login route
        });
    }

    private handleError(error: HttpErrorResponse) {
        console.error('Error occurred:', error);
        return throwError(() => new Error('Error in AuthService'));
    }
}
