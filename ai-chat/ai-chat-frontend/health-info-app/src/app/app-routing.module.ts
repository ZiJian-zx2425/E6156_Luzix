import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { ChatComponent } from './chat/chat.component'; // Import your ChatComponent

const routes: Routes = [
  { path: 'chat', component: ChatComponent },  // Route to the ChatComponent
  { path: '', redirectTo: '/chat', pathMatch: 'full' }, // Default route to ChatComponent
  // Define additional routes here
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
